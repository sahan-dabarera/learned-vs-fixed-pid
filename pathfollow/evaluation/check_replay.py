"""Check that re-run episode rows reproduce earlier rows bit for bit.

Compares two episodes.csv files on every (controller, seed, scenario) present
in both, over all columns except the inference-latency columns.

    python -m pathfollow.evaluation.check_replay NEW/episodes.csv OLD/episodes.csv --output NEW/replay_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathfollow.metrics import read_csv


def _key(row):
    return row["controller"], row["training_seed"], row["scenario_id"]


def _strip(row):
    return {k: v for k, v in row.items() if "inference" not in k}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new")
    parser.add_argument("old")
    parser.add_argument("--output")
    args = parser.parse_args()
    old = {_key(row): _strip(row) for row in read_csv(Path(args.old))}
    checked = identical = 0
    differing = []
    for row in read_csv(Path(args.new)):
        reference = old.get(_key(row))
        if reference is None:
            continue
        checked += 1
        if _strip(row) == reference:
            identical += 1
        elif len(differing) < 20:
            differing.append(list(_key(row)))
    result = {"new": args.new, "old": args.old, "rows_checked": checked, "bit_identical_rows": identical,
              "first_differing": differing}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"replay check: {checked} rows checked, {identical} bit-identical")
    if checked == 0 or identical != checked:
        raise SystemExit("replay check FAILED")


if __name__ == "__main__":
    main()
