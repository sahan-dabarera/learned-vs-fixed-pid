from dataclasses import asdict
import subprocess
import sys

import pytest

from pathfollow import config
from pathfollow.evaluation.check_reference import compare
from pathfollow.evaluation.make_test_scenarios import build_sweep, level_label, load_bookkeeping, save
from pathfollow.scenarios import ScenarioManifest, build_manifest


@pytest.fixture(scope="module")
def sealed():
    return build_manifest("validation")


def _key(scenario):
    data = asdict(scenario)
    for field in ("scenario_id", "initial_noise_std_m", "initial_delay_s", "events"):
        data.pop(field)
    return repr(sorted(data.items()))


def test_noise_sweep_overrides_only_the_noise_severity(sealed):
    manifest, book = build_sweep(sealed, "noise", [0.0003, 0.01], "t")
    base = {s.scenario_id: s for s in sealed.scenarios}
    for scenario in manifest.scenarios:
        record = base[book[scenario.scenario_id]["base_scenario_id"]]
        assert record.condition in {"noise", "combined"}
        assert _key(scenario) == _key(record)
        assert scenario.noise_seed == record.noise_seed
        assert scenario.initial_delay_s == record.initial_delay_s
        level = book[scenario.scenario_id]["level"]
        if scenario.evaluation_mode == "stationary":
            assert scenario.initial_noise_std_m == level
            assert scenario.events == record.events
        else:
            assert scenario.initial_noise_std_m == 0.0
            kinds = {event.kind: event for event in scenario.events}
            assert kinds["noise_step"].value == level
            assert kinds["noise_step"].start_s == config.CONFIG.transient_start_s
            if "delay_step" in kinds:
                assert kinds["delay_step"].value == config.CONFIG.delay_severity_s


def test_only_the_sealed_level_is_flagged_identical(sealed):
    _, book = build_sweep(sealed, "noise", [0.0003, 0.01], "t")
    for item in book.values():
        assert item["sealed_identical"] == (item["level"] == config.CONFIG.noise_severity_m)
    count = sum(item["sealed_identical"] for item in book.values())
    assert count == sum(s.condition in {"noise", "combined"} for s in sealed.scenarios)


def test_delay_sweep_covers_both_modes_and_keeps_noise_off(sealed):
    manifest, book = build_sweep(sealed, "delay", [0.0, 0.15, 0.3], "t")
    modes = {s.evaluation_mode for s in manifest.scenarios}
    assert modes == {"stationary", "transient"}
    for scenario in manifest.scenarios:
        assert scenario.initial_noise_std_m == 0.0
        assert all(event.kind == "delay_step" for event in scenario.events)
    assert sum(item["sealed_identical"] for item in book.values()) == sum(
        s.condition == "delay" for s in sealed.scenarios
    )


def test_sweep_manifest_round_trips_and_stays_loadable(sealed, tmp_path):
    manifest, book = build_sweep(sealed, "delay", [0.06], "t")
    target = tmp_path / "t.json"
    save(manifest, book, target, {"sweep": "delay"})
    assert ScenarioManifest.load(target).digest() == manifest.digest()
    assert load_bookkeeping(target) == book


def test_level_labels_are_identifier_safe():
    assert level_label("noise", 0.0003) == "0p3mm"
    assert level_label("delay", 0.15) == "0p15s"


def test_sweep_rejects_bad_levels(sealed):
    with pytest.raises(ValueError):
        build_sweep(sealed, "noise", [0.001, 0.001], "t")
    with pytest.raises(ValueError):
        build_sweep(sealed, "noise", [-1.0], "t")


def test_evaluator_refuses_the_sealed_directories():
    for sealed_dir in ("data/results/sealed", "data/results/sealed/sub"):
        result = subprocess.run(
            [sys.executable, "-m", "pathfollow.evaluation.evaluate", "--manifest", "data/scenarios/held_out.json",
             "--output", sealed_dir, "--no-default", "--robust-tau-d", "0.02"],
            cwd=config.ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "sealed" in result.stderr


def test_sealed_check_flags_completion_and_ignores_failed_padding():
    sealed_rows = [
        {"controller": "a", "training_seed": "", "scenario_id": "s1", "finished": "True",
         "failure_adjusted_error_m": "0.010", "mean_distance_m": "0.01", "wheel_saturation_fraction": "0"},
        {"controller": "a", "training_seed": "", "scenario_id": "s2", "finished": "False",
         "failure_adjusted_error_m": "0.500", "mean_distance_m": "0.2", "wheel_saturation_fraction": "0"},
    ]
    replayed = [
        {**sealed_rows[0], "scenario_id": "s1-noise-x", "base_scenario_id": "s1", "sweep": "noise",
         "sealed_identical": "True", "failure_adjusted_error_m": "0.011"},
        {**sealed_rows[1], "scenario_id": "s2-noise-x", "base_scenario_id": "s2", "sweep": "noise",
         "sealed_identical": "True", "failure_adjusted_error_m": "0.900"},
        {**sealed_rows[0], "scenario_id": "s1-noise-y", "base_scenario_id": "s1", "sweep": "noise",
         "sealed_identical": "False", "finished": "False"},
    ]
    result = compare(replayed, sealed_rows)
    assert result["rows_checked"] == 2
    assert result["completion_mismatches"] == []
    assert result["max_abs_diff_finished_rows"]["failure_adjusted_error_m"] == pytest.approx(0.001)


def test_env_defaults_keep_the_sealed_observation_sizes():
    from pathfollow.env import PathFollowingEnv

    assert PathFollowingEnv(mode="scheduled", training=False).observation_space.shape == (170,)
    assert PathFollowingEnv(mode="direct", training=False).observation_space.shape == (130,)


def test_history_length_one_shrinks_the_observation_to_one_frame():
    from pathfollow.env import PathFollowingEnv

    for mode, frame in (("scheduled", 17), ("direct", 13)):
        env = PathFollowingEnv(mode=mode, training=False, history_length=1)
        observation, _ = env.reset(seed=0, options=build_manifest("validation").scenarios[0].to_options())
        assert observation.shape == (frame,)
        assert env.observation_space.shape == (frame,)


def test_default_noise_draws_are_unchanged_by_the_later_options():
    from pathfollow.env import PathFollowingEnv

    def episodes(**kwargs):
        env = PathFollowingEnv(mode="scheduled", training=True, disturbance_training=True, **kwargs)
        env.reset(seed=7)
        return [env._sample_episode({}) for _ in range(50)]

    assert repr(episodes()) == repr(episodes(train_noise_log_range_m=None))


def test_log_uniform_noise_range_bounds_every_draw():
    from pathfollow.env import PathFollowingEnv

    env = PathFollowingEnv(
        mode="direct", training=True, disturbance_training=True,
        train_noise_log_range_m=(0.0003, 0.01),
    )
    env.reset(seed=3)
    values = []
    for _ in range(400):
        episode = env._sample_episode({})
        if episode["initial_noise_std_m"]:
            values.append(episode["initial_noise_std_m"])
        values.extend(e["value"] for e in episode["events"] if e["kind"] == "noise_step" and e["value"])
    assert values
    assert min(values) >= 0.0003 and max(values) <= 0.01
    assert max(values) > 0.003


def test_pid_retune_injects_sigma_only_into_noise_scenarios():
    from pathfollow.tuning.tune_pid_rn import with_noise
    from pathfollow.tuning.calibrate_pid import _subset

    for scenario in _subset(build_manifest("validation"), robust=True):
        noisy = with_noise(scenario, 0.01)
        if scenario.condition in {"noise", "combined"}:
            carried = noisy.initial_noise_std_m or max(
                e.value for e in noisy.events if e.kind == "noise_step"
            )
            assert carried == 0.01
        else:
            assert noisy == scenario


def test_fine_level_rule_brackets_the_first_break_and_triggers_e4():
    from pathfollow.evaluation.fine_noise_levels import breaking_points, cells, fine_levels

    def row(arm, level, finished, e_fa):
        return {"controller": arm, "condition": "noise", "level": str(level),
                "finished": str(finished), "failure_adjusted_error_m": str(e_fa)}

    rows = [
        row("ppo_direct_disturbed", 0.0003, True, 0.002),
        row("ppo_direct_disturbed", 0.001, True, 0.003),
        row("ppo_direct_disturbed", 0.003, True, 0.005),  # > 2x baseline
        row("pid_global_robust", 0.0003, True, 0.002),
        row("pid_global_robust", 0.001, True, 0.002),
        row("pid_global_robust", 0.003, True, 0.002),
    ]
    points = breaking_points(cells(rows))
    assert points[("ppo_direct_disturbed", "noise")]["bracket_m"] == (0.001, 0.003)
    assert points[("pid_global_robust", "noise")]["breaks_at_m"] is None
    levels = fine_levels(points)
    assert len(levels) == 3 and all(0.001 < value < 0.003 for value in levels)


def test_training_refuses_the_stored_checkpoints():
    result = subprocess.run(
        [sys.executable, "-m", "pathfollow.training.train", "--mode", "direct", "--regime", "nominal",
         "--arm", "ppo_direct_guardtest", "--output", str(config.CHECKPOINTS)],
        cwd=config.ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "refusing to train into the stored checkpoints" in result.stderr
    assert not (config.CHECKPOINTS / "ppo_direct_guardtest").exists()
