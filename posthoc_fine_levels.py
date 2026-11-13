"""Apply the plan's automatic rules to the coarse noise sweep.

Breaking point, per arm and condition (seed-mean, both evaluation modes
pooled, as the plan's 60-scenario cells are pooled over paths and speeds):
the first coarse level where completion < 100 % or E_FA > 2 x the arm's own
0.3 mm value. The fine levels are three log-spaced values strictly between the
bracketing coarse levels, union over arms, de-duplicated, at most six.

E4 trigger: GS-D or DR-D breaks at a level <= 30 mm in either condition.
sigma* is the lowest such level.

    python posthoc_fine_levels.py artifacts/results/posthoc/e1_noise_coarse
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from metrics import read_csv


ANCHOR_M = 0.0003
TRIGGER_MAX_M = 0.030
TRIGGER_ARMS = ("ppo_scheduler_disturbed", "ppo_direct_disturbed")
MAX_FINE = 6


def cells(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[float, dict[str, float]]]:
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["controller"], row["condition"], float(row["level"]))].append(row)
    table: dict[tuple[str, str], dict[float, dict[str, float]]] = defaultdict(dict)
    for (arm, condition, level), group in grouped.items():
        table[(arm, condition)][level] = {
            "completion": float(np.mean([row["finished"] == "True" for row in group])),
            "e_fa": float(np.mean([float(row["failure_adjusted_error_m"]) for row in group])),
            "episodes": len(group),
        }
    return table


def breaking_points(table) -> dict[tuple[str, str], dict]:
    result = {}
    for key, by_level in sorted(table.items()):
        levels = sorted(by_level)
        baseline = by_level[ANCHOR_M]["e_fa"]
        broken = None
        for level in levels:
            cell = by_level[level]
            if cell["completion"] < 1.0 or cell["e_fa"] > 2.0 * baseline:
                broken = level
                break
        bracket = None
        if broken is not None and broken != levels[0]:
            bracket = (levels[levels.index(broken) - 1], broken)
        result[key] = {"breaks_at_m": broken, "bracket_m": bracket, "baseline_e_fa_m": baseline}
    return result


def fine_levels(points: dict) -> list[float]:
    candidates = set()
    for item in points.values():
        if item["bracket_m"]:
            low, high = item["bracket_m"]
            inner = np.geomspace(low, high, 5)[1:-1]
            candidates.update(round(float(value), 6) for value in inner)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    rows = read_csv(Path(args.result_dir) / "episodes.csv")
    points = breaking_points(cells(rows))
    for (arm, condition), item in points.items():
        at = "never" if item["breaks_at_m"] is None else f"{item['breaks_at_m'] * 1000:g} mm"
        print(f"{arm:28s} {condition:9s} breaks at {at:9s} (baseline E_FA {item['baseline_e_fa_m'] * 1000:.2f} mm)")
    levels = fine_levels(points)
    print(f"\nfine candidates ({len(levels)}): {[f'{v * 1000:.3g}' for v in levels]} mm")
    if len(levels) > MAX_FINE:
        print(f"  more than {MAX_FINE}: the plan caps the fine pass, a choice is needed")
    if all(item["breaks_at_m"] is None for item in points.values()):
        print("  no arm breaks even at the top level: the plan adds 300 mm once")
    triggers = [
        item["breaks_at_m"] for (arm, _), item in points.items()
        if arm in TRIGGER_ARMS and item["breaks_at_m"] is not None and item["breaks_at_m"] <= TRIGGER_MAX_M
    ]
    sigma_star = min(triggers) if triggers else None
    print(f"\nE4 trigger: {'yes, sigma* = %g mm' % (sigma_star * 1000) if sigma_star else 'no'}")
    out = Path(args.result_dir) / "fine_levels.json"
    out.write_text(json.dumps({
        "breaking_points": {f"{a}|{c}": v for (a, c), v in points.items()},
        "fine_candidates_m": levels,
        "sigma_star_m": sigma_star,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
