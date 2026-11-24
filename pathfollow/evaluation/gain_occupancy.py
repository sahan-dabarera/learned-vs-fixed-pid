"""Gain occupancy: where in its gain box a scheduler checkpoint actually operates.

Replays scheduler checkpoints (deterministic actions) on one or more manifests
and records the target and applied (Kp, Ki, Kd) at every policy step. The box
and rate limit of each checkpoint come from its training metadata
(evaluate.model_box), so wide-box arms are measured against their own box.

Edge occupancy is measured in the policy's action coordinates, because the
gain map is piecewise linear around g0 and the two segments differ by a factor
of up to ~14 in Kp: a target counts as "at the lower edge" when its action
coordinate is <= -0.98 (2 % of the lower segment) and "at the upper edge" when
>= +0.98. The fraction of steps whose applied gain sits exactly on a bound is
reported too.

    python -m pathfollow.evaluation.gain_occupancy --models-root data/checkpoints \\
        --arm ppo_scheduler_ranged_h1=data/scenarios/validation_ranged.json \\
        --arm ppo_scheduler_ranged_h1=data/scenarios/held_out_ranged.json \\
        --output runs/results/occupancy
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from pathfollow import config
from pathfollow.metrics import write_csv
from pathfollow.evaluation.evaluate import git_state, model_box, usable_cores

GAINS = ("kp", "ki", "kd")
EDGE = 0.98


def _replay(arm: str, seed: int, model_path: str, manifest_path: str, calibration_path: str,
            rate_limit) -> dict[str, Any]:
    import torch
    from stable_baselines3 import PPO

    from pathfollow.calibration import GainCalibration, PIDGains
    from pathfollow.controller_profiles import profile_for
    from pathfollow.env import PathFollowingEnv
    from pathfollow.rollout import run_episode
    from pathfollow.scenarios import ScenarioManifest

    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    history = int(model.observation_space.shape[0]) // len(profile_for("scheduled").frame_fields)
    calibration = GainCalibration.load(Path(calibration_path))
    env = PathFollowingEnv(mode="scheduled", training=False, calibration=calibration, history_length=history,
                           **({"gain_rate_limit": tuple(rate_limit)} if rate_limit else {}))
    lower, upper = calibration.lower.as_array(), calibration.upper.as_array()
    per_condition: dict[str, dict[str, list]] = {}
    try:
        for scenario in ScenarioManifest.load(Path(manifest_path)).scenarios:
            trace = run_episode(env, scenario, lambda obs: model.predict(obs, deterministic=True)[0]).trace
            bucket = per_condition.setdefault(f"{scenario.evaluation_mode}/{scenario.condition}",
                                              {"applied": [], "target": []})
            bucket["applied"] += [[row[g] for g in GAINS] for row in trace]
            bucket["target"] += [[row[f"{g}_target"] for g in GAINS] for row in trace]
    finally:
        env.close()
    rows, samples = [], {}
    for condition, data in sorted(per_condition.items()):
        applied = np.asarray(data["applied"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        action = np.asarray([calibration.action_for(PIDGains(*g)) for g in target])
        row = {"controller": arm, "training_seed": seed, "manifest": Path(manifest_path).stem,
               "condition": condition, "policy_steps": len(applied),
               "calibration": Path(calibration_path).name}
        for i, g in enumerate(GAINS):
            row[f"{g}_target_at_lower_edge"] = float(np.mean(action[:, i] <= -EDGE))
            row[f"{g}_target_at_upper_edge"] = float(np.mean(action[:, i] >= EDGE))
            row[f"{g}_applied_on_lower_bound"] = float(np.mean(np.isclose(applied[:, i], lower[i])))
            row[f"{g}_applied_on_upper_bound"] = float(np.mean(np.isclose(applied[:, i], upper[i])))
            for name, value in (("mean", np.mean(applied[:, i])), ("median", np.median(applied[:, i])),
                                ("p5", np.percentile(applied[:, i], 5)), ("p95", np.percentile(applied[:, i], 95))):
                row[f"{g}_applied_{name}"] = float(value)
        rows.append(row)
        samples[condition] = applied[:, 0].astype(np.float32)
    return {"rows": rows, "kp_samples": samples, "arm": arm, "seed": seed, "manifest": Path(manifest_path).stem}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="ARM=MANIFEST")
    parser.add_argument("--scheduler-calibration", default=str(config.CALIBRATION / "blind_ppo.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", help="comma-separated seeds (default: the five final seeds)")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if (output / "occupancy.csv").exists():
        parser.error(f"refusing to overwrite {output / 'occupancy.csv'}")
    jobs = []
    for entry in args.arm:
        arm, manifest = entry.split("=", 1)
        for seed in (tuple(int(v) for v in args.seeds.split(",")) if args.seeds
                     else config.CONFIG.final_training_seeds):
            path = Path(args.models_root).resolve() / arm / f"seed_{seed}" / "best_model.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            box, rate = model_box(path, str(Path(args.scheduler_calibration).resolve()))
            jobs.append((arm, seed, str(path), str(Path(manifest).resolve()), box, rate))

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers or min(usable_cores(), len(jobs))) as pool:
        results = list(pool.map(_replay, *zip(*jobs)))

    from pathfollow.evaluation.evaluate_reference import machine_description

    output.mkdir(parents=True, exist_ok=True)
    rows = [row for item in results for row in item["rows"]]
    write_csv(output / "occupancy.csv", rows)
    np.savez_compressed(output / "kp_samples.npz", **{
        f"{item['arm']}|{item['manifest']}|{item['seed']}|{condition}": values
        for item in results for condition, values in item["kp_samples"].items()
    })
    (output / "metadata.json").write_text(json.dumps({
        "description": "Applied/target scheduler gains per policy step; edges in action coordinates (|a| >= 0.98)",
        "command": [sys.executable, *sys.argv], "git": git_state(), "machine": machine_description(),
        "wall_time_s": time.time() - started, "jobs": len(jobs),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
