"""Replay controllers with per-step traces (thesis trajectory and gain-over-time figures).

Runs the main thesis controllers on a manifest with the evaluation rollout of
evaluate.py (same environment, deterministic actions, reset seed 0) and
writes:

    episodes.csv      one row per episode: the usual episode metrics plus
                      mean / RMS / max |e_ct| over the policy steps.
                      e_ct is the signed cross-track error to the continuous
                      path, not the sampled-point distance used in E_FA.
    traces.csv.gz     per-step rows (t, x, y, e_ct, delay, noise, gains, ...)
                      for the scenarios matching --trace (fnmatch patterns).
    replay_check.json E_FA / finished / duration of every episode compared with
                      an earlier episodes.csv (--reference), must be identical.

Learned arms are given as ARM=MODELS_ROOT:SEED,SEED,... so old and new seeds
can come from different model roots.

    python -m pathfollow.evaluation.traces --manifest data/scenarios/held_out_ranged.json \\
        --arm ppo_direct_ranged_h1=data/checkpoints:11,23,37,53,71 \\
        --trace 'held_out-random_curvature_10001-v0.5-*' --output runs/results/traces_heldout
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from fnmatch import fnmatch
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from pathfollow import config
from pathfollow.calibration import GainCalibration
from pathfollow.metrics import read_csv, write_csv
from pathfollow.evaluation.evaluate import ControllerSpec, _scheduler_spec, _worker_init, git_state, usable_cores
from pathfollow.scenarios import Scenario, ScenarioManifest

TRACE_FIELDS = ("t", "x", "y", "e_ct", "e_ct_measured", "distance_m", "progress_m", "speed_mps",
                "actuator_delay_s", "sensor_noise_m", "omega_requested", "omega_applied",
                "kp", "ki", "kd", "kp_target", "ki_target", "kd_target")
PIDRN_GAINS = (2.0, 0.5, 0.5, 0.01)  # data/results/stage2/pid_retune_wide_fine/pid_rn.json "selected"


def build_specs(args) -> list[ControllerSpec]:
    calibration = str(Path(args.calibration).resolve())
    scheduler_calibration = str(Path(args.scheduler_calibration).resolve())
    pid = GainCalibration.load(Path(calibration))
    specs = [
        ControllerSpec("pid_global_nominal", None, "fixed", calibration, tuple(pid.nominal.as_array())),
        ControllerSpec("pid_global_robust", None, "fixed", calibration, tuple(pid.robust.as_array())),
        ControllerSpec("pid_global_robust_noise", None, "fixed", calibration, PIDRN_GAINS[:3],
                       derivative_filter_tau_s=PIDRN_GAINS[3]),
    ]
    for entry in args.arm:
        arm, rest = entry.split("=", 1)
        root, seeds = rest.rsplit(":", 1)
        for seed in (int(s) for s in seeds.split(",")):
            path = Path(root).resolve() / arm / f"seed_{seed}" / "best_model.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            specs.append(_scheduler_spec(arm, seed, path, calibration, scheduler_calibration))
    return specs


def _run(spec: ControllerSpec, scenarios: list[dict[str, Any]], patterns: list[str]):
    from pathfollow.evaluation.evaluate import _controller
    from pathfollow.rollout import run_episode

    env, predictor = _controller(spec)
    rows, traces = [], []
    for raw in scenarios:
        scenario = Scenario.from_dict(raw)
        result = run_episode(env, scenario, predictor)
        e_ct = np.asarray([row["e_ct"] for row in result.trace], dtype=np.float64)
        seed = "" if spec.seed is None else spec.seed
        rows.append({
            "controller": spec.name, "training_seed": seed, "scenario_id": scenario.scenario_id,
            "evaluation_mode": scenario.evaluation_mode, "condition": scenario.condition,
            "target_speed": scenario.target_speed,
            "mean_abs_e_ct_m": float(np.mean(np.abs(e_ct))),
            "rms_e_ct_m": float(np.sqrt(np.mean(e_ct ** 2))),
            "max_abs_e_ct_m": float(np.max(np.abs(e_ct))),
            "policy_steps": len(e_ct),
            **{k: v for k, v in result.metrics.items() if "inference" not in k},
        })
        if any(fnmatch(scenario.scenario_id, p) for p in patterns):
            for step, row in enumerate(result.trace):
                traces.append({"controller": spec.name, "training_seed": seed,
                               "scenario_id": scenario.scenario_id, "step": step,
                               **{k: row.get(k, "") for k in TRACE_FIELDS}})
    return rows, traces


def replay_check(rows, references: list[str]) -> dict[str, Any]:
    """Compare with earlier rows. References are merged in order and a later file
    overrides an earlier one on the same (controller, seed, scenario), so a
    superseded arm (an older PID-RN under the same name) is replaced by listing
    its final directory last."""
    ref = {}
    for reference in references:
        for r in read_csv(Path(reference)):
            ref[(r["controller"], str(r["training_seed"]), r["scenario_id"])] = r
    checked = identical = 0
    differing = []
    for row in rows:
        old = ref.get((row["controller"], str(row["training_seed"]), row["scenario_id"]))
        if old is None:
            continue
        checked += 1
        same = all(str(row[k]) == str(old[k]) for k in ("failure_adjusted_error_m", "finished", "duration_s"))
        identical += same
        if not same and len(differing) < 20:
            differing.append([row["controller"], row["training_seed"], row["scenario_id"]])
    return {"references": references, "rows_checked": checked, "unmatched_rows": len(rows) - checked,
            "identical_rows": identical,
            "first_differing": differing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="ARM=MODELS_ROOT:SEEDS")
    parser.add_argument("--only", action="append", default=[],
                        help="run only scenarios matching these fnmatch patterns (default: all)")
    parser.add_argument("--trace", action="append", default=[], help="fnmatch patterns of scenarios to trace")
    parser.add_argument("--no-pid", action="store_true", help="skip the three fixed PIDs")
    parser.add_argument("--reference", action="append", default=[], help="earlier episodes.csv for the replay check")
    parser.add_argument("--calibration", default=str(config.CALIBRATION / "calibration.json"))
    parser.add_argument("--scheduler-calibration",
                        default=str(config.CALIBRATION / "blind_ppo.json"))
    parser.add_argument("--warn-only", action="store_true",
                        help="report but do not fail on replay-check differences (provisional runs off the cluster)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=12)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if (output / "episodes.csv").exists():
        parser.error(f"refusing to overwrite {output / 'episodes.csv'}")
    specs = build_specs(args)
    if args.no_pid:
        specs = [s for s in specs if s.mode != "fixed"]
    manifest = ScenarioManifest.load(Path(args.manifest).resolve())
    scenarios = [asdict(s) for s in manifest.scenarios
                 if not args.only or any(fnmatch(s.scenario_id, p) for p in args.only)]
    if not scenarios:
        parser.error("no scenarios selected")
    tasks = [(spec, scenarios[i:i + args.chunk]) for spec in specs for i in range(0, len(scenarios), args.chunk)]
    started = time.time()
    workers = args.workers or min(usable_cores(), len(tasks))
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for r, t in pool.map(_run, *zip(*[(s, c, args.trace) for s, c in tasks])):
            rows += r
            traces += t

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "episodes.csv", rows)
    if traces:
        with gzip.open(output / "traces.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
            writer.writeheader()
            writer.writerows(traces)
    checks = [replay_check(rows, args.reference)] if args.reference else []
    from pathfollow.evaluation.evaluate_reference import machine_description

    (output / "metadata.json").write_text(json.dumps({
        "description": "Replays with per-episode |e_ct| statistics and per-step traces (pathfollow.evaluation.traces)",
        "manifest": str(Path(args.manifest).resolve()), "trace_patterns": args.trace, "only": args.only,
        "controllers": [[s.name, s.seed] for s in specs], "episodes": len(rows), "trace_rows": len(traces),
        "replay_checks": checks, "command": [sys.executable, *sys.argv], "git": git_state(),
        "machine": machine_description(), "wall_time_s": time.time() - started,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} episodes, {len(traces)} trace rows in {(time.time() - started) / 60:.1f} min -> {output}")
    for check in checks:
        print(f"replay check: {check['rows_checked']} rows checked ({check['unmatched_rows']} without a reference), "
              f"{check['identical_rows']} identical")
        failed = check["rows_checked"] == 0 or check["identical_rows"] != check["rows_checked"] or check["unmatched_rows"]
        if failed and args.warn_only:
            print("replay check: rows differ from the reference (allowed with --warn-only: provisional run on "
                  "another machine)")
        elif failed:
            raise SystemExit("replay check FAILED")


if __name__ == "__main__":
    main()
