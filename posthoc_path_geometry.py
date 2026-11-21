"""Export the reference paths of the training, validation and held-out manifests
as x, y points (thesis figure W17a). Pure geometry: no simulation, no controller.

Every distinct path in a manifest is built with core.paths.build, exactly as the
environment builds it, and written once. Paths start at the origin heading +x,
which is also the robot's initial pose in every scenario.

    python posthoc_path_geometry.py --output artifacts/results/posthoc4/w17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
from core import paths
from metrics import write_csv

MANIFESTS = {
    "train": config.ROOT / "manifests" / "train.json",
    "validation": config.ROOT / "manifests" / "validation.json",
    "held_out": config.ROOT / "manifests" / "posthoc" / "held_out_ranged.json",
}


def path_id(spec) -> str:
    if isinstance(spec, str):
        return spec
    params = spec.get("params", {})
    return f"{spec['kind']}_{params['seed']}" if "seed" in params else spec["kind"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=5, help="keep every N-th path point (spacing is 1 cm)")
    args = parser.parse_args()

    points, summary = [], []
    for split, manifest in MANIFESTS.items():
        specs = {}
        for scenario in json.loads(manifest.read_text(encoding="utf-8"))["scenarios"]:
            specs.setdefault(path_id(scenario["path"]), scenario["path"])
        for pid, spec in sorted(specs.items()):
            path = paths.build(spec)
            kappa = paths.curvature_of(path)
            keep = np.r_[np.arange(0, len(path["s"]), args.step), len(path["s"]) - 1]
            keep = np.unique(keep)
            for i in keep:
                points.append({"split": split, "path_id": pid, "s_m": float(path["s"][i]),
                               "x_m": float(path["pts"][i, 0]), "y_m": float(path["pts"][i, 1]),
                               "curvature_per_m": float(kappa[i])})
            summary.append({"split": split, "path_id": pid, "name": path["name"],
                            "spec": json.dumps(spec, sort_keys=True), **paths.geometry(path)})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "path_points.csv", points)
    write_csv(output / "path_summary.csv", summary)
    print(f"{len(summary)} paths, {len(points)} points -> {output}")


if __name__ == "__main__":
    main()
