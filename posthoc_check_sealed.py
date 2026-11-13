"""Compare post-hoc rows that replay a sealed scenario against the sealed run.

A post-hoc sweep row flagged ``sealed_identical`` runs a scenario identical to
a sealed held-out record (only the identifier differs), with the same
controller instance. Its outcome must reproduce the sealed ``episodes.csv``:
completion exactly, continuous metrics to within drift.

    python posthoc_check_sealed.py artifacts/results/posthoc/e1_noise_coarse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from metrics import read_csv


SEALED_EPISODES = config.ROOT / "artifacts" / "results" / "final" / "episodes.csv"
METRICS = ("failure_adjusted_error_m", "mean_distance_m", "wheel_saturation_fraction")


def _flag(value: str) -> bool:
    return str(value).lower() == "true"


def compare(posthoc_rows: list[dict[str, str]], sealed_rows: list[dict[str, str]]) -> dict:
    sealed = {(row["controller"], row["training_seed"], row["scenario_id"]): row for row in sealed_rows}
    checked = 0
    missing = 0
    completion_mismatch = []
    worst: dict[str, float] = {metric: 0.0 for metric in METRICS}
    worst_relative: dict[str, float] = {metric: 0.0 for metric in METRICS}
    exact = 0
    for row in posthoc_rows:
        if not _flag(row.get("sealed_identical", "")) and row.get("sweep"):
            continue
        key = (row["controller"], row["training_seed"], row["base_scenario_id"])
        reference = sealed.get(key)
        if reference is None:
            missing += 1
            continue
        checked += 1
        if _flag(row["finished"]) != _flag(reference["finished"]):
            completion_mismatch.append({"controller": key[0], "seed": key[1], "scenario": key[2]})
        identical = True
        both_finished = _flag(row["finished"]) and _flag(reference["finished"])
        for metric in METRICS:
            a, b = float(row[metric]), float(reference[metric])
            difference = abs(a - b)
            identical &= difference == 0.0
            if not both_finished:
                # A failed episode's E_FA is dominated by exit-time padding,
                # so a millisecond of exit drift moves it by percent.
                continue
            worst[metric] = max(worst[metric], difference)
            if b != 0:
                worst_relative[metric] = max(worst_relative[metric], difference / abs(b))
        exact += identical
    return {
        "rows_checked": checked,
        "rows_without_sealed_counterpart": missing,
        "bit_identical_rows": exact,
        "completion_mismatches": completion_mismatch,
        "max_abs_diff_finished_rows": worst,
        "max_rel_diff_finished_rows": worst_relative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--sealed", default=str(SEALED_EPISODES))
    args = parser.parse_args()
    sealed_rows = read_csv(Path(args.sealed))
    report = {}
    for raw in args.result_dirs:
        directory = Path(raw).resolve()
        result = compare(read_csv(directory / "episodes.csv"), sealed_rows)
        report[directory.name] = result
        (directory / "sealed_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"{directory.name}: {result['rows_checked']} rows checked, "
            f"{result['bit_identical_rows']} bit-identical, "
            f"{len(result['completion_mismatches'])} completion mismatches, "
            f"max |dE_FA| on finished rows {result['max_abs_diff_finished_rows']['failure_adjusted_error_m'] * 1000:.4f} mm "
            f"({result['max_rel_diff_finished_rows']['failure_adjusted_error_m'] * 100:.2f} %)"
        )


if __name__ == "__main__":
    main()
