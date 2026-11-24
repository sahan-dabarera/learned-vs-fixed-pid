"""Validation scores of the selected checkpoint, per arm and seed.

Training writes one ``validation_history.jsonl`` line per checkpoint
evaluation and keeps the checkpoint with the best (completion, -E_FA), first
one winning a tie (train.py ManifestEvalCallback). This script re-applies that
rule to the logged history, so no replay is needed.

    python -m pathfollow.evaluation.validation_scores --output runs/results/validation \\
        --arm ppo_scheduler_disturbed=data/checkpoints \\
        --arm ppo_scheduler_disturbed_h1=data/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathfollow import config
from pathfollow.metrics import write_csv


def selected_checkpoint(history_path: Path) -> dict:
    best, best_key = None, None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["completion_rate"], -record["failure_adjusted_error_m"])
        if best_key is None or key > best_key:
            best, best_key = record, key
    if best is None:
        raise ValueError(f"{history_path} is empty")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, metavar="ARM=MODELS_ROOT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", help="comma-separated seeds (default: the five final seeds)")
    args = parser.parse_args()
    seeds = (tuple(int(value) for value in args.seeds.split(",")) if args.seeds
             else config.CONFIG.final_training_seeds)

    rows = []
    for entry in args.arm:
        arm, root = entry.split("=", 1)
        for seed in seeds:
            run_dir = Path(root).resolve() / arm / f"seed_{seed}"
            record = selected_checkpoint(run_dir / "validation_history.jsonl")
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            rows.append({
                "controller": arm,
                "seed": seed,
                "selected_timesteps": record["timesteps"],
                "completion_rate": record["completion_rate"],
                "failure_adjusted_error_m": record["failure_adjusted_error_m"],
                "episodes": record["episodes"],
                "validation_manifest": Path(metadata["validation_manifest"]).name,
                "history_length": (metadata.get("posthoc") or {}).get("history_length") or 10,
            })
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "validation_selected.csv", rows)
    for row in rows:
        print(f"{row['controller']:40s} seed {row['seed']}  completion {row['completion_rate']:.3f}  "
              f"E_FA {row['failure_adjusted_error_m'] * 1000:.2f} mm  ({row['validation_manifest']})")


if __name__ == "__main__":
    main()
