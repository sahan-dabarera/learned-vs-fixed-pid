"""Parallel paired evaluation for the post-hoc experiments.

The episode loop is the sealed one: the same ``PathFollowingEnv``, the same
``run_episode``, deterministic policy actions, and ``reset_seed=0``. What this
script adds is parallelism across cores, extra controller variants, and sweep
bookkeeping in the output rows. It never writes into the sealed result or
model directories.

Controller sets (combine freely):

    default         PID-N, PID-R, and the 20 sealed learned checkpoints
    --frozen-gains  constant-gain scheduler variants from posthoc_frozen_gains.py
    --robust-tau-d  PID-R re-run with a different derivative filter constant

A frozen variant runs in scheduler mode with a constant action that maps to the
frozen gains, so it starts from the nominal gains and is rate-limited towards
its target exactly as the learned scheduler is; only the target stops moving.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from calibration import GainCalibration, PIDGains
import config
from metrics import write_csv
from posthoc_manifests import load_bookkeeping
from scenarios import Scenario, ScenarioManifest


LEARNED_ARMS = (
    "ppo_scheduler_nominal",
    "ppo_scheduler_disturbed",
    "ppo_direct_nominal",
    "ppo_direct_disturbed",
)
SEALED_DIRS = (
    config.ROOT / "artifacts" / "results" / "final",
    config.ROOT / "artifacts" / "models" / "final",
)
SUMMARY_METRICS = (
    "failure_adjusted_error_m",
    "mean_distance_m",
    "max_distance_m",
    "steer_total_variation_per_s",
    "wheel_saturation_fraction",
    "speed_ratio",
    "gain_total_variation_per_s",
    "post_event_peak_distance_m",
    "post_event_excess_iae_m_s",
)


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    seed: int | None
    mode: str  # fixed | scheduled | direct
    calibration_path: str
    fixed_gains: tuple[float, float, float] | None = None
    model_path: str | None = None
    constant_gains: tuple[float, float, float] | None = None
    derivative_filter_tau_s: float | None = None
    history_length: int | None = None


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def usable_cores() -> int:
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit():
        return int(allocated)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def build_controllers(args: argparse.Namespace) -> list[ControllerSpec]:
    calibration = str(Path(args.calibration).resolve())
    scheduler_calibration = str(Path(args.scheduler_calibration).resolve())
    pid = GainCalibration.load(Path(calibration))
    specs: list[ControllerSpec] = []
    if not args.no_default:
        specs.append(ControllerSpec("pid_global_nominal", None, "fixed", calibration, tuple(pid.nominal.as_array())))
        specs.append(ControllerSpec("pid_global_robust", None, "fixed", calibration, tuple(pid.robust.as_array())))
        models_root = Path(args.models_root).resolve()
        for arm in LEARNED_ARMS:
            if args.arms and arm not in args.arms:
                continue
            for seed in config.CONFIG.final_training_seeds:
                path = models_root / arm / f"seed_{seed}" / "best_model.zip"
                if not path.is_file():
                    raise FileNotFoundError(path)
                mode = "scheduled" if arm.startswith("ppo_scheduler_") else "direct"
                specs.append(ControllerSpec(
                    arm, seed, mode,
                    scheduler_calibration if mode == "scheduled" else calibration,
                    model_path=str(path),
                ))
    for tau in args.robust_tau_d:
        specs.append(ControllerSpec(
            f"pid_global_robust_taud{_label(tau)}", None, "fixed", calibration,
            tuple(pid.robust.as_array()), derivative_filter_tau_s=float(tau),
        ))
    if args.frozen_gains:
        payload = json.loads(Path(args.frozen_gains).read_text(encoding="utf-8"))
        statistic = args.frozen_statistic
        for arm, seeds in sorted(payload["gains"].items()):
            for seed, variants in sorted(seeds.items(), key=lambda item: int(item[0])):
                for variant in args.frozen_variants:
                    gains = variants[variant][statistic]
                    specs.append(ControllerSpec(
                        f"{arm}_frozen_{variant}", int(seed), "scheduled", scheduler_calibration,
                        constant_gains=(gains["kp"], gains["ki"], gains["kd"]),
                    ))
    for entry in args.model:
        identity, raw_path = entry.split("=", 1)
        arm, raw_seed = identity.rsplit(":", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mode = "scheduled" if arm.startswith("ppo_scheduler_") else "direct"
        specs.append(ControllerSpec(
            arm, int(raw_seed), mode,
            scheduler_calibration if mode == "scheduled" else calibration,
            model_path=str(path),
        ))
    names = [(spec.name, spec.seed) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate controller/seed in the requested set")
    if not specs:
        raise ValueError("controller set is empty")
    return specs


# ---------------------------------------------------------------- worker side

_CACHE: dict[ControllerSpec, tuple[Any, Any]] = {}


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _controller(spec: ControllerSpec) -> tuple[Any, Any]:
    if spec in _CACHE:
        return _CACHE[spec]
    from env import PathFollowingEnv

    calibration = GainCalibration.load(Path(spec.calibration_path))
    model = None
    history_length = spec.history_length
    if spec.model_path is not None:
        from stable_baselines3 import PPO

        model = PPO.load(spec.model_path, device="cpu")
    kwargs: dict[str, Any] = {}
    if history_length is not None:
        kwargs["history_length"] = history_length
    env = PathFollowingEnv(
        mode=spec.mode,
        training=False,
        calibration=calibration,
        fixed_gains=PIDGains(*spec.fixed_gains) if spec.fixed_gains else None,
        derivative_filter_tau_s=spec.derivative_filter_tau_s,
        **kwargs,
    )
    if model is not None:
        predictor = lambda obs, loaded=model: loaded.predict(obs, deterministic=True)[0]
    elif spec.constant_gains is not None:
        action = calibration.action_for(PIDGains(*spec.constant_gains))
        predictor = lambda obs, fixed=action: fixed
    else:
        predictor = None
    _CACHE[spec] = (env, predictor)
    return env, predictor


def _run_chunk(spec: ControllerSpec, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from rollout import run_episode

    env, predictor = _controller(spec)
    rows = []
    for raw in scenarios:
        scenario = Scenario.from_dict(raw)
        result = run_episode(env, scenario, predictor)
        rows.append({
            "controller": spec.name,
            "training_seed": "" if spec.seed is None else spec.seed,
            "evaluation_mode": scenario.evaluation_mode,
            "condition": scenario.condition,
            "path_kind": scenario.path["kind"],
            "target_speed": scenario.target_speed,
            "initial_delay_s": scenario.initial_delay_s,
            "initial_noise_std_m": scenario.initial_noise_std_m,
            **result.metrics,
            "speed_ratio": float(result.metrics["mean_speed_mps"]) / scenario.target_speed,
        })
    return rows


# ---------------------------------------------------------------- aggregation

def _mean(values: list[Any]) -> float | str:
    numbers = [float(value) for value in values if value not in (None, "")]
    return float(np.mean(numbers)) if numbers else ""


def summarize(rows: list[dict[str, Any]], by_seed: bool) -> list[dict[str, Any]]:
    keys = ["controller", "training_seed", "manifest", "sweep", "level", "level_label", "evaluation_mode", "condition"]
    if not by_seed:
        keys.remove("training_seed")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key))
        record["episodes"] = len(group)
        record["seeds"] = len({str(row["training_seed"]) for row in group})
        record["completion_rate"] = float(np.mean([bool(row["finished"]) for row in group]))
        for metric in SUMMARY_METRICS:
            record[metric] = _mean([row.get(metric) for row in group])
        output.append(record)
    return output


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True, help="e.g. artifacts/results/posthoc/e1_noise_coarse")
    parser.add_argument("--calibration", default=str(config.ROOT / "artifacts" / "calibration" / "calibration.json"))
    parser.add_argument("--scheduler-calibration", default=str(config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"))
    parser.add_argument("--models-root", default=str(config.ROOT / "artifacts" / "models" / "final"))
    parser.add_argument("--arms", nargs="*", default=[], help="restrict the default learned arms")
    parser.add_argument("--no-default", action="store_true", help="skip PID-N, PID-R and the sealed checkpoints")
    parser.add_argument("--model", action="append", default=[], metavar="ARM:SEED=PATH")
    parser.add_argument("--robust-tau-d", type=float, action="append", default=[])
    parser.add_argument("--frozen-gains")
    parser.add_argument("--frozen-variants", nargs="+", default=["clean", "delay"])
    parser.add_argument("--frozen-statistic", choices=("mean", "median"), default="mean")
    parser.add_argument("--condition", action="append", default=[], help="restrict scenarios to these conditions")
    parser.add_argument("--mode", action="append", default=[], help="restrict scenarios to these evaluation modes")
    parser.add_argument("--workers", type=int, default=0, help="0 uses the granted cores")
    parser.add_argument("--chunk", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N scenarios per manifest")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    for sealed in SEALED_DIRS:
        if output == sealed.resolve() or sealed.resolve() in output.parents:
            parser.error(f"refusing to write inside the sealed directory {sealed}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty result directory: {output}")

    specs = build_controllers(args)
    manifests = []
    for raw in args.manifest:
        path = Path(raw).resolve()
        manifest = ScenarioManifest.load(path)
        scenarios = [
            scenario for scenario in manifest.scenarios
            if (not args.condition or scenario.condition in args.condition)
            and (not args.mode or scenario.evaluation_mode in args.mode)
        ]
        if args.limit:
            scenarios = scenarios[: args.limit]
        manifests.append((path, manifest, scenarios, load_bookkeeping(path)))

    tasks = []
    for spec in specs:
        for path, _, scenarios, _ in manifests:
            payload = [asdict(scenario) for scenario in scenarios]
            for start in range(0, len(payload), args.chunk):
                tasks.append((spec, payload[start:start + args.chunk], path))
    total = sum(len(chunk) for _, chunk, _ in tasks)
    workers = args.workers or usable_cores()
    scenario_ids = [scenario.scenario_id for _, _, scenarios, _ in manifests for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        parser.error("scenario identifiers repeat across the given manifests")
    print(f"controllers: {len(specs)}  scenarios: {len(scenario_ids)}"
          f"  episodes: {total}  workers: {workers}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    started = time.time()
    rows: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [(pool.submit(_run_chunk, spec, chunk), path) for spec, chunk, path in tasks]
        for future, path in futures:
            chunk_rows = future.result()
            for row in chunk_rows:
                row["manifest"] = path.stem
            rows.extend(chunk_rows)
            done += len(chunk_rows)
            if done % max(total // 20, 1) < len(chunk_rows):
                elapsed = time.time() - started
                print(f"  {done}/{total} episodes, {elapsed / 60:.1f} min", flush=True)

    bookkeeping = {}
    for _, _, _, book in manifests:
        bookkeeping.update(book)
    for row in rows:
        book = bookkeeping.get(str(row["scenario_id"]), {})
        row["sweep"] = book.get("sweep", "")
        row["level"] = book.get("level", "")
        row["level_label"] = book.get("level_label", "")
        row["base_scenario_id"] = book.get("base_scenario_id", row["scenario_id"])
        row["sealed_identical"] = book.get("sealed_identical", "")

    expected = {
        (spec.name, "" if spec.seed is None else str(spec.seed), scenario_id)
        for spec in specs
        for scenario_id in scenario_ids
    }
    actual = {(str(row["controller"]), str(row["training_seed"]), str(row["scenario_id"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("controller/scenario result matrix is incomplete or duplicated")

    leading = ["controller", "training_seed", "manifest", "sweep", "level", "level_label",
               "evaluation_mode", "condition", "scenario_id", "base_scenario_id", "sealed_identical"]
    rows = [{**{key: row[key] for key in leading}, **row} for row in rows]
    rows.sort(key=lambda row: (row["controller"], str(row["training_seed"]), row["manifest"], row["scenario_id"]))
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summarize(rows, by_seed=False))
    write_csv(output / "summary_by_seed.csv", summarize(rows, by_seed=True))
    wall = time.time() - started

    from evaluate import machine_description

    metadata = {
        "command": [sys.executable, *sys.argv],
        "git": git_state(),
        "machine": machine_description(),
        "workers": workers,
        "wall_time_s": wall,
        "episodes": len(rows),
        "manifests": [
            {"path": str(path), "sha256_scenarios": manifest.digest(), "evaluated": len(scenarios)}
            for path, manifest, scenarios, _ in manifests
        ],
        "filters": {"condition": args.condition, "mode": args.mode, "limit": args.limit},
        "calibration": {"path": args.calibration, "sha256": sha256(Path(args.calibration))},
        "scheduler_calibration": {"path": args.scheduler_calibration, "sha256": sha256(Path(args.scheduler_calibration))},
        "controllers": [
            {**asdict(spec), "model_sha256": sha256(Path(spec.model_path)) if spec.model_path else None}
            for spec in specs
        ],
        "frozen_gains": args.frozen_gains,
        "frozen_statistic": args.frozen_statistic if args.frozen_gains else None,
        "config": config.CONFIG.to_dict(),
        "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} episodes in {wall / 60:.1f} min -> {output}")


if __name__ == "__main__":
    main()
