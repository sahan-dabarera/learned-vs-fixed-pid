"""E3: the constant gains each sealed scheduler checkpoint actually applies.

Replays every scheduler checkpoint on the VALIDATION manifest (never held-out)
and records the applied (Kp, Ki, Kd) at every policy step. Two operating
points per checkpoint:

    clean   validation scenarios with the nominal condition (both modes)
    delay   validation scenarios whose condition contains delay
            (delay and combined, both modes)

Steps are pooled across scenarios, so the mean is a time average. The applied
gains include the rate-limited ramp from the nominal gains at the start of
every episode, exactly as the frozen variant will experience it.

    python posthoc_frozen_gains.py --output artifacts/results/posthoc/e3_frozen_gains
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

import config
from posthoc_eval import git_state, model_box, usable_cores


SCHEDULER_ARMS = ("ppo_scheduler_nominal", "ppo_scheduler_disturbed")
VARIANTS = {
    "clean": ("nominal",),
    "delay": ("delay", "combined"),
}


def _replay(arm: str, seed: int, model_path: str, manifest_path: str, calibration_path: str,
            rate_limit: tuple[float, float, float] | None = None) -> dict[str, Any]:
    import torch
    from stable_baselines3 import PPO

    from calibration import GainCalibration
    from env import PathFollowingEnv
    from rollout import run_episode
    from scenarios import ScenarioManifest

    from controller_profiles import profile_for

    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    # An H=1 checkpoint has a shorter input; read the history off the network.
    history_length = int(model.observation_space.shape[0]) // len(profile_for("scheduled").frame_fields)
    env = PathFollowingEnv(
        mode="scheduled",
        training=False,
        calibration=GainCalibration.load(Path(calibration_path)),
        history_length=history_length,
        **({"gain_rate_limit": rate_limit} if rate_limit is not None else {}),
    )
    manifest = ScenarioManifest.load(Path(manifest_path))
    result: dict[str, Any] = {}
    try:
        for variant, conditions in VARIANTS.items():
            gains = []
            scenarios = [s for s in manifest.scenarios if s.condition in conditions]
            for scenario in scenarios:
                trace = run_episode(
                    env, scenario, lambda obs: model.predict(obs, deterministic=True)[0]
                ).trace
                gains.extend((row["kp"], row["ki"], row["kd"]) for row in trace)
            values = np.asarray(gains, dtype=np.float64)
            result[variant] = {
                "mean": dict(zip(("kp", "ki", "kd"), values.mean(axis=0).tolist())),
                "median": dict(zip(("kp", "ki", "kd"), np.median(values, axis=0).tolist())),
                "std": dict(zip(("kp", "ki", "kd"), values.std(axis=0).tolist())),
                "policy_steps": int(len(values)),
                "scenarios": len(scenarios),
                "conditions": list(conditions),
            }
    finally:
        env.close()
    return {"arm": arm, "seed": seed, "gains": result,
            "box": {"calibration": calibration_path, "gain_rate_limit_per_s": rate_limit}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=str(config.ROOT / "manifests" / "validation.json"))
    parser.add_argument("--scheduler-calibration", default=str(config.ROOT / "artifacts" / "calibration" / "blind_ppo.json"))
    parser.add_argument("--models-root", default=str(config.ROOT / "artifacts" / "models" / "final"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", help="comma-separated seeds (default: the five final seeds)")
    parser.add_argument("--arm", action="append", default=[],
                        help="scheduler arm under --models-root (default: the two sealed scheduler arms)")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    if not manifest.name.startswith("validation"):
        parser.error("frozen gains must come from a validation manifest")
    output = Path(args.output).resolve()
    target = output / "frozen_gains.json"
    if target.exists():
        parser.error(f"refusing to overwrite {target}")

    jobs = []
    for arm in args.arm or SCHEDULER_ARMS:
        for seed in (tuple(int(v) for v in args.seeds.split(",")) if args.seeds
                     else config.CONFIG.final_training_seeds):
            path = Path(args.models_root).resolve() / arm / f"seed_{seed}" / "best_model.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            box, rate = model_box(path, str(Path(args.scheduler_calibration).resolve()))
            jobs.append((arm, seed, str(path), str(manifest), box, rate))

    started = time.time()
    workers = args.workers or min(usable_cores(), len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_replay, *zip(*jobs)))

    gains: dict[str, dict[str, Any]] = {}
    boxes: dict[str, dict[str, Any]] = {}
    for item in results:
        gains.setdefault(item["arm"], {})[str(item["seed"])] = item["gains"]
        boxes.setdefault(item["arm"], {})[str(item["seed"])] = item["box"]
    from evaluate import machine_description

    payload = {
        "description": "Time-averaged applied scheduler gains on the validation manifest",
        "variants": {name: list(conditions) for name, conditions in VARIANTS.items()},
        "gains": gains,
        "boxes": boxes,
        "command": [sys.executable, *sys.argv],
        "git": git_state(),
        "machine": machine_description(),
        "wall_time_s": time.time() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for arm, seeds in gains.items():
        for seed, variants in seeds.items():
            line = "  ".join(
                f"{name}: Kp {v['mean']['kp']:.1f} Ki {v['mean']['ki']:.2f} Kd {v['mean']['kd']:.2f}"
                for name, v in variants.items()
            )
            print(f"{arm} seed {seed}  {line}")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
