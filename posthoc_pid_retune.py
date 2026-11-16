"""E4 fairness: re-tune the robust fixed PID for sigma* sensor noise (PID-RN).

The sealed robust PID (8.0, 0.5, 4.4) was not taken from differential
evolution: it came from a Kp sweep on the 28-scenario robust VALIDATION subset
(``calibrate_pid._subset(validation, robust=True)``), selected by the
practical-equivalence rule in ``calibrate_pid.select_practical_candidate``
(DECISIONS.md, 2026-08-23). PID-RN repeats that procedure with two changes:

* every noise-bearing scenario in the subset carries sigma* instead of 0.3 mm;
* the grid adds Kd and the derivative filter constant tau_d, the two knobs an
  engineer would reach for against measurement noise.

Ki stays at the sealed 0.5, as in the sealed sweep. The sealed PID-R point is
always in the grid, so the retune can only match or beat it on this subset.

    python posthoc_pid_retune.py --sigma-m 0.01 --output artifacts/results/posthoc/e4_pid_retune

Plan 2 runs it without --sigma-m on the ranged validation manifest, so every
scenario keeps its own drawn delay and noise:

    python posthoc_pid_retune.py --manifest manifests/posthoc/validation_ranged.json --output ...
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from calibrate_pid import _subset, select_practical_candidate
import config
from metrics import write_csv
from posthoc_eval import git_state, usable_cores
from scenarios import Scenario, ScenarioManifest


KP_GRID = (4.0, 6.0, 8.0, 10.0, 13.0, 18.0, 26.0, 40.0)
KD_GRID = (0.5, 1.0, 2.0, 3.0, 4.4, 6.0, 8.0)
TAU_GRID = (0.01, 0.02, 0.05, 0.1)
KI = 0.5
SEALED_ROBUST = (8.0, 0.5, 4.4, 0.01)


def with_noise(scenario: Scenario, sigma: float) -> Scenario:
    if scenario.condition not in {"noise", "combined"}:
        return scenario
    if scenario.evaluation_mode == "stationary":
        return replace(scenario, initial_noise_std_m=float(sigma))
    return replace(scenario, events=tuple(
        replace(event, value=float(sigma)) if event.kind == "noise_step" else event
        for event in scenario.events
    ))


def score(candidate: tuple[float, float, float, float], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    from calibration import PIDGains
    from env import PathFollowingEnv
    from rollout import run_episode

    kp, ki, kd, tau = candidate
    env = PathFollowingEnv(
        mode="fixed", training=False, fixed_gains=PIDGains(kp, ki, kd), derivative_filter_tau_s=tau
    )
    try:
        rows = [run_episode(env, Scenario.from_dict(raw)).metrics for raw in scenarios]
    finally:
        env.close()
    return {
        "kp": kp, "ki": ki, "kd": kd, "tau_d": tau,
        "completion_rate": float(np.mean([row["finished"] for row in rows])),
        "failure_adjusted_error_m": float(np.mean([row["failure_adjusted_error_m"] for row in rows])),
        "mean_distance_m": float(np.mean([row["mean_distance_m"] for row in rows])),
        "steer_total_variation_per_s": float(np.mean([row["steer_total_variation_per_s"] for row in rows])),
        "wheel_saturation_fraction": float(np.mean([row["wheel_saturation_fraction"] for row in rows])),
        "gain_l1": kp + ki + kd,
        "is_sealed_robust": candidate == SEALED_ROBUST,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-m", type=float, help="sigma* in metres; omit to keep the manifest's noise")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--kp-grid", help="comma-separated Kp values replacing the default grid")
    args = parser.parse_args()
    kp_grid = KP_GRID if args.kp_grid is None else tuple(float(v) for v in args.kp_grid.split(","))

    output = Path(args.output).resolve()
    if (output / "pid_rn.json").exists():
        parser.error(f"refusing to overwrite {output / 'pid_rn.json'}")
    manifest = ScenarioManifest.load(Path(args.manifest))
    subset = [
        scenario if args.sigma_m is None else with_noise(scenario, args.sigma_m)
        for scenario in _subset(manifest, robust=True)
    ]
    payload = [asdict(scenario) for scenario in subset]
    grid = sorted(set(itertools.product(kp_grid, (KI,), KD_GRID, TAU_GRID)) | {SEALED_ROBUST})

    started = time.time()
    workers = args.workers or usable_cores()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(score, grid, itertools.repeat(payload), chunksize=1))
    chosen = select_practical_candidate(rows)
    sealed = next(row for row in rows if row["is_sealed_robust"])

    from evaluate import machine_description

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidates.csv", rows)
    result = {
        "sigma_m": args.sigma_m,
        "manifest": Path(args.manifest).name,
        "manifest_sha256": manifest.digest(),
        "selected": {key: chosen[key] for key in ("kp", "ki", "kd", "tau_d")},
        "selected_metrics": chosen,
        "sealed_robust_metrics": sealed,
        "selection_rule": "calibrate_pid.select_practical_candidate (completion band, 0.5% error band, smoothest)",
        "scenarios": [scenario.scenario_id for scenario in subset],
        "grid": {"kp": kp_grid, "ki": KI, "kd": KD_GRID, "tau_d": TAU_GRID},
        "command": [sys.executable, *sys.argv],
        "git": git_state(),
        "machine": machine_description(),
        "wall_time_s": time.time() - started,
    }
    (output / "pid_rn.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    s = result["selected"]
    edges = [
        name for name, value, grid in (("kp", s["kp"], kp_grid), ("kd", s["kd"], KD_GRID), ("tau_d", s["tau_d"], TAU_GRID))
        if value in (min(grid), max(grid))
    ]
    result["selected_on_grid_edge"] = edges
    (output / "pid_rn.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if edges:
        print(f"WARNING: selection sits on the grid edge in {edges}; report it")
    where = "manifest noise" if args.sigma_m is None else f"sigma*={args.sigma_m * 1000:g} mm"
    print(f"PID-RN at {where}: Kp {s['kp']} Ki {s['ki']} Kd {s['kd']} tau_d {s['tau_d']}")
    print(f"  selected E_FA {chosen['failure_adjusted_error_m'] * 1000:.2f} mm, completion {chosen['completion_rate']:.2f}")
    print(f"  sealed PID-R E_FA {sealed['failure_adjusted_error_m'] * 1000:.2f} mm, completion {sealed['completion_rate']:.2f}")
    print(f"--pid pid_global_robust_noise={s['kp']},{s['ki']},{s['kd']},{s['tau_d']}")


if __name__ == "__main__":
    main()
