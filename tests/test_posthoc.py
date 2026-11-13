from dataclasses import asdict, replace
import subprocess
import sys

import pytest

import config
from posthoc_check_sealed import compare
from posthoc_manifests import build_sweep, level_label, load_bookkeeping, save
from scenarios import ScenarioManifest, build_manifest


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
    for sealed_dir in ("artifacts/results/final", "artifacts/results/final/sub"):
        result = subprocess.run(
            [sys.executable, "posthoc_eval.py", "--manifest", "manifests/held_out.json",
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
    posthoc = [
        {**sealed_rows[0], "scenario_id": "s1-noise-x", "base_scenario_id": "s1", "sweep": "noise",
         "sealed_identical": "True", "failure_adjusted_error_m": "0.011"},
        {**sealed_rows[1], "scenario_id": "s2-noise-x", "base_scenario_id": "s2", "sweep": "noise",
         "sealed_identical": "True", "failure_adjusted_error_m": "0.900"},
        {**sealed_rows[0], "scenario_id": "s1-noise-y", "base_scenario_id": "s1", "sweep": "noise",
         "sealed_identical": "False", "finished": "False"},
    ]
    result = compare(posthoc, sealed_rows)
    assert result["rows_checked"] == 2
    assert result["completion_mismatches"] == []
    assert result["max_abs_diff_finished_rows"]["failure_adjusted_error_m"] == pytest.approx(0.001)
