"""Build the post-hoc severity-sweep manifests from the sealed held-out manifest.

Every swept scenario is a copy of one sealed held-out record with exactly one
disturbance severity overridden. Path, speed, initial pose, event timing, and
noise seed are inherited unchanged, so every level of a sweep is paired with
every other level and with the sealed run.

    python posthoc_manifests.py noise --levels-mm 0.3,1,3,10,30,100 --name noise_coarse
    python posthoc_manifests.py delay --levels-s 0,0.03,0.06,0.10,0.15,0.20,0.30,0.40 --name delay

Noise sweep: the base records are the sealed ``noise`` and ``combined``
scenarios in both evaluation modes. Stationary records get the swept value as
their initial noise; transient records get it as the value of their noise step
at the sealed onset. The combined condition keeps its sealed 0.15 s delay.

Delay sweep: the base records are the sealed ``delay`` scenarios in both modes,
with noise off, exactly as sealed.

At the sealed severity a swept scenario is identical to its sealed record apart
from its identifier. Those rows are flagged ``sealed_identical`` so a sweep
doubles as a replay check of the sealed results.

The sweep bookkeeping lives in a top-level ``posthoc`` block, which
``ScenarioManifest.load`` ignores, so the files stay loadable by evaluate.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import config
from scenarios import DisturbanceEvent, Scenario, ScenarioManifest


SEALED_MANIFEST = config.ROOT / "manifests" / "held_out.json"
OUTPUT_DIR = config.ROOT / "manifests" / "posthoc"

SWEEPS: dict[str, dict[str, Any]] = {
    "noise": {
        "conditions": ("noise", "combined"),
        "event_kind": "noise_step",
        "initial_field": "initial_noise_std_m",
        "sealed_value": config.CONFIG.noise_severity_m,
    },
    "delay": {
        "conditions": ("delay",),
        "event_kind": "delay_step",
        "initial_field": "initial_delay_s",
        "sealed_value": config.CONFIG.delay_severity_s,
    },
}


def level_label(sweep: str, value: float) -> str:
    """Filesystem- and identifier-safe label, e.g. 0p3mm or 0p15s."""
    text = f"{value * 1000.0:g}mm" if sweep == "noise" else f"{value:g}s"
    return text.replace(".", "p")


def override_severity(scenario: Scenario, sweep: str, value: float) -> Scenario:
    spec = SWEEPS[sweep]
    if scenario.evaluation_mode == "stationary":
        return replace(scenario, **{spec["initial_field"]: float(value)})
    events = []
    found = False
    for event in scenario.events:
        if event.kind == spec["event_kind"]:
            event = DisturbanceEvent(event.kind, event.start_s, float(value), event.end_s)
            found = True
        events.append(event)
    if not found:
        raise ValueError(f"{scenario.scenario_id} has no {spec['event_kind']} to override")
    return replace(scenario, events=tuple(events))


def same_except_id(a: Scenario, b: Scenario) -> bool:
    left, right = asdict(a), asdict(b)
    left.pop("scenario_id")
    right.pop("scenario_id")
    return left == right


def build_sweep(
    sealed: ScenarioManifest,
    sweep: str,
    levels: list[float],
    name: str,
) -> tuple[ScenarioManifest, dict[str, dict[str, Any]]]:
    if sweep not in SWEEPS:
        raise ValueError(f"unknown sweep: {sweep}")
    if not levels or len(set(levels)) != len(levels) or any(value < 0 for value in levels):
        raise ValueError("levels must be unique and non-negative")
    conditions = SWEEPS[sweep]["conditions"]
    base = [scenario for scenario in sealed.scenarios if scenario.condition in conditions]
    scenarios: list[Scenario] = []
    bookkeeping: dict[str, dict[str, Any]] = {}
    for value in levels:
        label = level_label(sweep, value)
        for record in base:
            swept = override_severity(record, sweep, value)
            scenario_id = f"{record.scenario_id}-{sweep}-{label}"
            swept = replace(swept, scenario_id=scenario_id)
            scenarios.append(swept)
            bookkeeping[scenario_id] = {
                "sweep": sweep,
                "level": float(value),
                "level_label": label,
                "base_scenario_id": record.scenario_id,
                "sealed_identical": same_except_id(swept, record),
            }
    manifest = ScenarioManifest(name=f"posthoc_{name}", split=sealed.split, scenarios=tuple(scenarios))
    return manifest, bookkeeping


def save(manifest: ScenarioManifest, bookkeeping: dict[str, dict[str, Any]], path: Path, extra: dict[str, Any]) -> None:
    payload = {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "split": manifest.split,
        "posthoc": {"source": extra, "scenarios": bookkeeping},
        "scenarios": [asdict(scenario) for scenario in manifest.scenarios],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_bookkeeping(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("posthoc", {}).get("scenarios", {}))


def _floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", choices=tuple(SWEEPS))
    parser.add_argument("--levels-mm", help="noise standard deviations in millimetres")
    parser.add_argument("--levels-s", help="actuator delays in seconds")
    parser.add_argument("--name", required=True)
    parser.add_argument("--sealed", default=str(SEALED_MANIFEST))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.sweep == "noise":
        if not args.levels_mm:
            parser.error("noise sweep needs --levels-mm")
        levels = [value / 1000.0 for value in _floats(args.levels_mm)]
    else:
        if not args.levels_s:
            parser.error("delay sweep needs --levels-s")
        levels = _floats(args.levels_s)

    sealed_path = Path(args.sealed).resolve()
    sealed = ScenarioManifest.load(sealed_path)
    manifest, bookkeeping = build_sweep(sealed, args.sweep, levels, args.name)
    target = Path(args.output).resolve() / f"{args.name}.json"
    save(
        manifest,
        bookkeeping,
        target,
        {
            "sweep": args.sweep,
            "levels": levels,
            "sealed_manifest": sealed_path.name,
            "sealed_manifest_sha256": sealed.digest(),
        },
    )
    anchors = sum(1 for item in bookkeeping.values() if item["sealed_identical"])
    print(
        f"{args.name}: {len(manifest.scenarios)} scenarios, {len(levels)} levels, "
        f"{anchors} identical to sealed records, digest {manifest.digest()[:12]} -> {target}"
    )


if __name__ == "__main__":
    main()
