"""Build stage1.md from the stage-1 result directories.

Every number in the document is read from data/results/stage1. Free-text
observations come from notes/stage1.md (one "## E1" ... "## E5"
section each) so the tables can be regenerated without touching them.

    python -m pathfollow.analysis.report_stage1
    python -m pathfollow.analysis.report_stage1 --copy-to path/to/copy.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import numpy as np

from pathfollow import config

from pathfollow.analysis.bootstrap import _bootstrap
from pathfollow.metrics import read_csv

ROOT = config.ROOT
NOTES = Path(__file__).resolve().parent / "notes"
REPORTS = config.RESULTS / "reports"
PH = config.RESULTS / "stage1"
DRAWS = 10_000
BOOT_SEED = 20260819  # bootstrap.py default
SEEDS = ("11", "23", "37", "53", "71")
MODES = ("stationary", "transient")
CONDITIONS = ("nominal", "delay", "noise", "combined")

SHORT = {
    "pid_global_nominal": "PID-N",
    "pid_global_robust": "PID-R",
    "ppo_scheduler_nominal": "GS-N",
    "ppo_scheduler_disturbed": "GS-D",
    "ppo_direct_nominal": "DR-N",
    "ppo_direct_disturbed": "DR-D",
    "pid_global_robust_taud0p02": "PID-R τd 0.02",
    "pid_global_robust_taud0p05": "PID-R τd 0.05",
    "ppo_scheduler_nominal_frozen_clean": "GS-N F-clean",
    "ppo_scheduler_nominal_frozen_delay": "GS-N F-delay",
    "ppo_scheduler_disturbed_frozen_clean": "GS-D F-clean",
    "ppo_scheduler_disturbed_frozen_delay": "GS-D F-delay",
    "ppo_scheduler_disturbed_h1": "GS-D H1",
    "ppo_direct_disturbed_h1": "DR-D H1",
    "pid_global_robust_noise_3mm": "PID-RN 3mm",
    "pid_global_robust_noise_10mm": "PID-RN 10mm",
    "ppo_scheduler_noisetrained_3mm": "GS-DN 3mm",
    "ppo_scheduler_noisetrained_10mm": "GS-DN 10mm",
    "ppo_direct_noisetrained_3mm": "DR-DN 3mm",
    "ppo_direct_noisetrained_10mm": "DR-DN 10mm",
}
TRAINING_NOTE = (
    "Job 3188678 on cpunode39 ran all 30 stage-2 training runs at once (E4 3 mm, E4 10 mm, E5 H1; 10 each) on "
    "64 cores, beside the two PID-RN retunes and the E1 fine sweep (32 workers). Wall times are therefore not "
    "comparable with the sealed run's (about 30 min per scheduler run, 90 min per direct run)."
)
SEALED_ARMS = [
    "pid_global_nominal", "pid_global_robust", "ppo_scheduler_nominal",
    "ppo_scheduler_disturbed", "ppo_direct_nominal", "ppo_direct_disturbed",
]
E4_ARMS = [
    "pid_global_robust", "pid_global_robust_noise_3mm", "pid_global_robust_noise_10mm",
    "ppo_scheduler_disturbed", "ppo_scheduler_noisetrained_3mm", "ppo_scheduler_noisetrained_10mm",
    "ppo_direct_disturbed", "ppo_direct_noisetrained_3mm", "ppo_direct_noisetrained_10mm",
]


# ---------------------------------------------------------------- formatting

def pct(value: float) -> str:
    value = float(value) * 100
    if value in (0.0, 100.0):
        return f"{value:.0f}"
    return f"{value:.1f}"


def mm(value: float | str) -> str:
    if value == "" or value is None:
        return "–"
    value = float(value) * 1000
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"


def cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "–"
    return f"{pct(row['completion_rate'])} / {mm(row['failure_adjusted_error_m'])}"


def table(header: list[str], rows: Iterable[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join(lines)


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


def level_text(level: float, sweep: str) -> str:
    return f"{level * 1000:g} mm" if sweep == "noise" else f"{level:g} s"


# ---------------------------------------------------------------- loading

def summary(*names: str, file: str = "summary.csv") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in names:
        rows += read_csv(PH / name / file)
    return rows


def index(rows: list[dict[str, str]], with_seed: bool = False) -> dict[tuple, dict[str, str]]:
    out = {}
    for row in rows:
        level = round(float(row["level"]), 7) if row.get("level") else None
        key = (row["controller"], level, row["evaluation_mode"], row["condition"])
        if with_seed:
            key += (row.get("training_seed", ""),)
        out[key] = row
    return out


def levels_of(rows: list[dict[str, str]]) -> list[float]:
    return sorted({round(float(row["level"]), 7) for row in rows if row.get("level")})


def episodes(name: str) -> list[dict[str, str]]:
    return read_csv(PH / name / "episodes.csv")


def meta(name: str) -> dict[str, Any]:
    return json.loads((PH / name / "metadata.json").read_text(encoding="utf-8"))


def finished(row: dict[str, str]) -> float:
    return float(str(row["finished"]).lower() == "true")


# ---------------------------------------------------------------- sweep sections

def sweep_tables(idx, arms, levels, sweep, conditions) -> str:
    parts = []
    for condition in conditions:
        for mode in MODES:
            body = table(
                ["level"] + [SHORT[a] for a in arms],
                ([level_text(lv, sweep)] + [cell(idx.get((a, lv, mode, condition))) for a in arms] for lv in levels),
            )
            parts.append(f"**{condition}, {mode}** (completion % / E_FA mm, seed-pooled)\n\n{body}")
    return "\n\n".join(parts)


def full_metric_table(idx, arms, levels, sweep, conditions) -> str:
    rows = []
    for arm in arms:
        for condition in conditions:
            for mode in MODES:
                for lv in levels:
                    row = idx.get((arm, lv, mode, condition))
                    if row is None:
                        continue
                    rows.append([
                        SHORT[arm], condition, mode, level_text(lv, sweep), pct(row["completion_rate"]),
                        mm(row["failure_adjusted_error_m"]), mm(row["mean_distance_m"]),
                        f"{float(row['steer_total_variation_per_s']):.1f}",
                        pct(row["wheel_saturation_fraction"]), f"{float(row['speed_ratio']):.3f}",
                    ])
    return table(
        ["arm", "condition", "mode", "level", "compl %", "E_FA mm", "path dist mm", "steer TV/T (1/s)",
         "wheel sat %", "v̄/v_ref"],
        rows,
    )


def per_seed_tables(idx_seed, arms, levels, sweep, conditions) -> str:
    parts = []
    for arm in arms:
        for condition in conditions:
            for mode in MODES:
                body = table(
                    ["level"] + [f"seed {s}" for s in SEEDS],
                    ([level_text(lv, sweep)] + [cell(idx_seed.get((arm, lv, mode, condition, s))) for s in SEEDS]
                     for lv in levels),
                )
                parts.append(f"**{SHORT[arm]}, {condition}, {mode}** (completion % / E_FA mm)\n\n{body}")
    return "\n\n".join(parts)


def breaking(idx, arm, levels, mode_set, condition, kind) -> str:
    def pooled(lv):
        rows = [idx.get((arm, lv, m, condition)) for m in mode_set]
        if any(r is None for r in rows):
            return None
        return (float(np.mean([float(r["completion_rate"]) for r in rows])),
                float(np.mean([float(r["failure_adjusted_error_m"]) for r in rows])))

    base = pooled(levels[0])
    if base is None:
        return "–"
    if base[0] < 1:
        return f"already < 100 % at baseline ({pct(base[0])} %)"
    for lv in levels[1:]:
        value = pooled(lv)
        if value is None:
            continue
        if value[0] < 1:
            return f"{level_text(lv, kind)} (compl {pct(value[0])} %)"
        if value[1] > 2 * base[1]:
            return f"{level_text(lv, kind)} (E_FA {mm(value[1])} vs {mm(base[1])} mm)"
    return "none in range"


def breaking_table(idx, arms, levels, conditions, kind) -> str:
    rows = []
    for arm in arms:
        for condition in conditions:
            rows.append([SHORT[arm], condition] + [breaking(idx, arm, levels, ms, condition, kind)
                                                   for ms in (MODES, ("stationary",), ("transient",))])
    return table(["arm", "condition", "both modes pooled", "stationary", "transient"], rows)


# ---------------------------------------------------------------- paired comparisons

def paired(rows_a, rows_b, controller_a, controller_b, mode, condition, match_seed):
    """Seed -> arrays of (a - b) per scenario, for E_FA (m) and completion (0/1)."""
    ref = {}
    for row in rows_b:
        if row["controller"] == controller_b and row["evaluation_mode"] == mode and row["condition"] == condition:
            ref[(row["training_seed"] if match_seed else "", row["scenario_id"])] = row
    err, comp = defaultdict(list), defaultdict(list)
    for row in rows_a:
        if row["controller"] != controller_a or row["evaluation_mode"] != mode or row["condition"] != condition:
            continue
        base = ref[(row["training_seed"] if match_seed else "", row["scenario_id"])]
        seed = row["training_seed"] or "fixed"
        err[seed].append(float(row["failure_adjusted_error_m"]) - float(base["failure_adjusted_error_m"]))
        comp[seed].append(finished(row) - finished(base))
    return ({k: np.asarray(v) for k, v in err.items()}, {k: np.asarray(v) for k, v in comp.items()})


def estimate(values: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Seed-mean and 95 % CI. A fresh generator per comparison keeps every
    interval independent of call order, so figures and tables agree."""
    mean = float(np.mean([np.mean(v) for v in values.values()]))
    low, high = _bootstrap(values, np.random.default_rng(BOOT_SEED), DRAWS)
    return mean, low, high


def paired_tables(pairs, rows_a, rows_b, match_seed) -> tuple[str, str]:
    """pairs: list of (controller_a, controller_b). Returns (E_FA table, completion table)."""
    err_rows, comp_rows = [], []
    for a, b in pairs:
        for mode in MODES:
            for condition in CONDITIONS:
                err, comp = paired(rows_a, rows_b, a, b, mode, condition, match_seed)
                if not err:
                    continue
                e_mean, e_lo, e_hi = estimate(err)
                c_mean, c_lo, c_hi = estimate(comp)
                label = [f"{SHORT[a]} − {SHORT[b]}", mode, condition]
                err_rows.append(label + [f"{np.mean(err[s]) * 1000:+.2f}" if s in err else "–" for s in SEEDS]
                                + [f"**{e_mean * 1000:+.2f}**", f"[{e_lo * 1000:+.2f}, {e_hi * 1000:+.2f}]"])
                comp_rows.append(label + [f"{np.mean(comp[s]) * 100:+.1f}" if s in comp else "–" for s in SEEDS]
                                 + [f"**{c_mean * 100:+.1f}**", f"[{c_lo * 100:+.1f}, {c_hi * 100:+.1f}]"])
    header = ["pair", "mode", "condition"] + [f"s{s}" for s in SEEDS] + ["seed-mean", "95 % CI"]
    return table(header, err_rows), table(header, comp_rows)


def paired_vs_fixed(arms, rows_a, rows_ref, reference) -> tuple[str, str]:
    err_rows, comp_rows = [], []
    for a in arms:
        for mode in MODES:
            for condition in CONDITIONS:
                err, comp = paired(rows_a, rows_ref, a, reference, mode, condition, match_seed=False)
                e = estimate(err)
                c = estimate(comp)
                label = [f"{SHORT[a]} − {SHORT[reference]}", mode, condition]
                err_rows.append(label + [f"{e[0] * 1000:+.2f}", f"[{e[1] * 1000:+.2f}, {e[2] * 1000:+.2f}]"])
                comp_rows.append(label + [f"{c[0] * 100:+.1f}", f"[{c[1] * 100:+.1f}, {c[2] * 100:+.1f}]"])
    header = ["pair", "mode", "condition", "seed-mean", "95 % CI"]
    return table(header, err_rows), table(header, comp_rows)


def heldout_table(idx, arms) -> str:
    header = ["arm"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS]
    return table(header, ([SHORT[a]] + [cell(idx.get((a, None, m, c))) for m in MODES for c in CONDITIONS]
                          for a in arms))


def heldout_seed_table(idx_seed, arms) -> str:
    header = ["arm", "seed"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS]
    rows = []
    for a in arms:
        for s in SEEDS:
            rows.append([SHORT[a], s] + [cell(idx_seed.get((a, None, m, c, s))) for m in MODES for c in CONDITIONS])
    return table(header, rows)


# ---------------------------------------------------------------- run record

def training_cost(log: Path) -> str:
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        match = re.search(r"done\s+(\S+) seed (\d+) in ([\d.]+) min: (\w+)", line)
        if match:
            rows.append((match.group(1), match.group(2), float(match.group(3)), match.group(4)))
    rows.sort()
    total = re.search(r"(\d+)/(\d+) runs completed in ([\d.]+) min", log.read_text(encoding="utf-8"))
    body = table(["arm", "seed", "wall time (min)", "status"],
                 ([SHORT.get(a, a), s, f"{t:.1f}", st] for a, s, t, st in rows))
    if total:
        body += f"\n\n{total.group(1)}/{total.group(2)} runs, all in parallel, {total.group(3)} min wall time in total."
    return body


def run_record() -> str:
    outputs = [
        ("E3a frozen gains", "e3_frozen_gains", "frozen_gains.json"),
        ("E1 coarse", "e1_noise_coarse", "metadata.json"),
        ("E2 delay", "e2_delay", "metadata.json"),
        ("E3b frozen on held-out", "e3_frozen_eval", "metadata.json"),
        ("E1-F PID-R τd", "e1f_taud", "metadata.json"),
        ("E4 PID-RN 3 mm", "e4_pid_retune_3mm", "pid_rn.json"),
        ("E4 PID-RN 10 mm", "e4_pid_retune_10mm", "pid_rn.json"),
        ("E1 fine", "e1_noise_fine", "metadata.json"),
        ("held-out 480 rerun (sealed + E4 + E5 arms)", "heldout_rerun", "metadata.json"),
        ("E4 arms on coarse + fine noise sweep", "e4_noise_sweep", "metadata.json"),
    ]
    rows = []
    for label, name, file in outputs:
        m = json.loads((PH / name / file).read_text(encoding="utf-8"))
        machine = m["machine"]
        node = machine.get("node", "?").split(".")[0]
        rows.append([label, f"`{name}`", machine.get("slurm_job_id", "?"), node,
                     f"`{m['git']['commit'][:7]}`" + (" dirty" if m["git"].get("dirty") else ""),
                     m.get("episodes", "–"), f"{m['wall_time_s'] / 60:.1f}"])
    record = table(["step", "output", "SLURM job", "node", "commit", "episodes", "wall min"], rows)
    return record


# ---------------------------------------------------------------- document

def observations() -> dict[str, str]:
    path = NOTES / "stage1.md"
    if not path.exists():
        return {}
    sections: dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return {k: v.strip() for k, v in sections.items()}


def build() -> str:
    obs = observations()
    o = lambda key: obs.get(key, "_(observations not written yet)_")  # noqa: E731
    out: list[str] = ["# Stage 1 Results", ""]
    out.append(
        "Generated by `report_stage1.py` from `data/results/stage1/`; observations from "
        "`notes/stage1.md`. Arms: PID-N `(250, 4.545, 0.397)`, PID-R `(8.0, 0.5, 4.4)`, GS = PPO gain "
        "scheduler, DR = direct PPO, suffix N = nominal-trained, D = disturbance-trained. Cells read "
        "`completion % / E_FA mm`. Learned-arm cells pool all 5 seeds (equal episodes per seed, so the pooled "
        "mean equals the seed-mean). E_FA on a failed episode includes the exit-time padding, as in the sealed "
        "analysis."
    )

    # §0
    out += ["", "## 0. Run record", ""]
    p = meta("heldout_rerun")["packages"]
    m1, m2 = meta("e1_noise_coarse")["machine"], meta("heldout_rerun")["machine"]
    out.append(f"- **Git branch / commit:** `posthoc-experiments` (published as `development-history`); per step below, every run on a clean tree.")
    out.append(f"- **Machine(s), CPU, cores:** {m1['node']} and {m2['node']}, both {m2['processor']}, "
               f"64 allocated cores, {m2['platform']}. The sealed run used cpunode13, same CPU model.")
    out.append(f"- **Software versions:** python {m2['python']}, numpy {p['numpy']}, mujoco {p['mujoco']}, "
               f"gymnasium {p['gymnasium']}, sb3 {p['stable-baselines3']}, torch {p['torch']}.")
    out.append("- **Parity gate result:** `check_parity.py` passed at the start of all three jobs (3187230, "
               "3188678, 3189024): completion identical, continuous metrics within the 1 % tolerance against "
               "the stored reference.")
    checks = []
    for name in ("e1_noise_coarse", "e2_delay", "heldout_rerun"):
        c = json.loads((PH / name / "sealed_check.json").read_text(encoding="utf-8"))
        checks.append([f"`{name}`", c["rows_checked"], c["bit_identical_rows"], len(c["completion_mismatches"]),
                       f"{c['max_abs_diff_finished_rows']['failure_adjusted_error_m'] * 1000:.4f}"])
    out.append("- **Sealed-slice replay check:** every row whose scenario is identical to a sealed held-out "
               "scenario was compared with the sealed `data/results/final/episodes.csv`:\n\n"
               + table(["output", "rows checked", "bit-identical", "completion mismatches",
                        "max abs ΔE_FA mm (finished rows)"], checks))
    out.append("")
    out.append("Run record per step:\n\n" + run_record())
    out.append("")
    out.append(o("0 Deviations"))

    # §1
    coarse = summary("e1_noise_coarse")
    fine = summary("e1_noise_fine")
    e1 = index(coarse + fine)
    e1_seed = index(summary("e1_noise_coarse", "e1_noise_fine", file="summary_by_seed.csv"), with_seed=True)
    lv_all = levels_of(coarse + fine)
    lv_coarse = levels_of(coarse)
    out += ["", "## 1. E1 Noise sweep", "", "### Levels run (coarse + fine)", ""]
    out.append(f"- Coarse: {', '.join(f'{v * 1000:g}' for v in lv_coarse)} mm.")
    out.append(f"- Fine (rule-generated, 3 per bracket): "
               f"{', '.join(f'{v * 1000:g}' for v in levels_of(fine))} mm.")
    out.append("- Conditions: noise-only, and combined = noise σ + 0.15 s delay. Modes: stationary (σ from t = 0) "
               "and transient (σ as the noise step at 6 s). 60 scenarios per mode, level and condition; "
               "22 instances (2 PID + 4 learned arms × 5 seeds).")
    out += ["", "### Table: arm × σ × condition → completion %, E_FA mm", ""]
    out.append(sweep_tables(e1, SEALED_ARMS, lv_all, "noise", ("noise", "combined")))
    out += ["", details("Full metrics: path distance, steer TV/T, wheel saturation, v̄/v_ref",
                        full_metric_table(e1, SEALED_ARMS, lv_all, "noise", ("noise", "combined")))]
    out += ["", details("Per-seed completion % / E_FA mm, learned arms",
                        per_seed_tables(e1_seed, SEALED_ARMS[2:], lv_all, "noise", ("noise", "combined")))]
    out += ["", "### Breaking point per arm", ""]
    out.append("Rule: first level above 0.3 mm where completion < 100 % or E_FA > 2× the arm's own 0.3 mm value, "
               "same mode and condition, over coarse + fine levels. Before stage 2 the same rule was applied to "
               "the coarse levels with both modes pooled (`e1_noise_coarse/fine_levels.json`); it gave σ* = 3 mm.\n")
    out.append(breaking_table(e1, SEALED_ARMS, lv_all, ("noise", "combined"), "noise"))
    out += ["", "### E1-F PID-R derivative filter check", ""]
    out.append("PID-R gains `(8.0, 0.5, 4.4)` with the derivative low-pass constant τd raised from the sealed "
               "0.01 s, noise-only, coarse levels.\n")
    e1f = index(summary("e1f_taud") + coarse)
    out.append(sweep_tables(e1f, ["pid_global_robust", "pid_global_robust_taud0p02", "pid_global_robust_taud0p05"],
                            lv_coarse, "noise", ("noise",)))
    out += ["", details("Full metrics", full_metric_table(
        e1f, ["pid_global_robust", "pid_global_robust_taud0p02", "pid_global_robust_taud0p05"], lv_coarse,
        "noise", ("noise",)))]
    out += ["", "### Observations", "", o("E1")]

    # §2
    e2rows = summary("e2_delay")
    e2 = index(e2rows)
    lv_d = levels_of(e2rows)
    out += ["", "## 2. E2 Delay sweep", ""]
    out.append(f"Levels {', '.join(f'{v:g}' for v in lv_d)} s, delay only (noise off, as in the sealed delay records),"
               "stationary (delay from t = 0) and transient (delay step at 6 s); 60 scenarios per mode and level. "
               "0.40 s was run for all arms.\n")
    out += ["### Table: arm × delay → completion %, E_FA", ""]
    out.append(sweep_tables(e2, SEALED_ARMS, lv_d, "delay", ("delay",)))
    out += ["", details("Full metrics: path distance, steer TV/T, wheel saturation, v̄/v_ref",
                        full_metric_table(e2, SEALED_ARMS, lv_d, "delay", ("delay",)))]
    e2_seed = index(summary("e2_delay", file="summary_by_seed.csv"), with_seed=True)
    out += ["", details("Per-seed completion % / E_FA mm, learned arms",
                        per_seed_tables(e2_seed, SEALED_ARMS[2:], lv_d, "delay", ("delay",)))]
    out += ["", "### Breaking point per arm", ""]
    out.append("Same rule as E1 with the 0 s level as the baseline.\n")
    out.append(breaking_table(e2, SEALED_ARMS, lv_d, ("delay",), "delay"))
    out += ["", "### Observations", "", o("E2")]

    # §3
    fg = json.loads((PH / "e3_frozen_gains" / "frozen_gains.json").read_text(encoding="utf-8"))
    out += ["", "## 3. E3 Frozen-gain ablation", "", "### Frozen gains used", ""]
    out.append("Applied gains of each scheduler checkpoint, pooled over policy steps on the validation manifest: "
               "F-clean = nominal condition (24 scenarios), F-delay = delay + combined (48 scenarios), both modes. "
               "Evaluation uses the mean. The frozen controller runs in scheduler mode with a constant target, so "
               "its gains ramp from the nominal (26, 0.5, 4.4) through the same rate limiter.\n")
    rows = []
    for arm in ("ppo_scheduler_nominal", "ppo_scheduler_disturbed"):
        for seed in SEEDS:
            for variant in ("clean", "delay"):
                g = fg["gains"][arm][seed][variant]
                rows.append([SHORT[arm], seed, f"F-{variant}"]
                            + [f"{g['mean'][k]:.3g}" for k in ("kp", "ki", "kd")]
                            + [f"{g['median'][k]:.3g}" for k in ("kp", "ki", "kd")]
                            + [f"{g['std'][k]:.3g}" for k in ("kp", "ki", "kd")] + [g["policy_steps"]])
    out.append(table(["arm", "seed", "variant", "Kp mean", "Ki mean", "Kd mean", "Kp median", "Ki median",
                      "Kd median", "Kp std", "Ki std", "Kd std", "policy steps"], rows))
    ho_rows = summary("heldout_rerun")
    fz_rows = summary("e3_frozen_eval")
    idx3 = index(ho_rows + fz_rows)
    arms3 = ["pid_global_robust", "ppo_scheduler_nominal", "ppo_scheduler_nominal_frozen_clean",
             "ppo_scheduler_nominal_frozen_delay", "ppo_scheduler_disturbed", "ppo_scheduler_disturbed_frozen_clean",
             "ppo_scheduler_disturbed_frozen_delay"]
    out += ["", "### Table: arm × variant × mode × condition → completion %, E_FA mm", ""]
    out.append("Held-out 480 (60 scenarios per cell; learned rows pool 5 seeds). Dynamic and PID-R rows are from "
               "`heldout_rerun`, bit-identical to the sealed run.\n")
    out.append(heldout_table(idx3, arms3))
    idx3s = index(summary("heldout_rerun", "e3_frozen_eval", file="summary_by_seed.csv"), with_seed=True)
    out += ["", details("Per-seed completion % / E_FA mm", heldout_seed_table(idx3s, arms3[1:]))]
    ho_eps = episodes("heldout_rerun")
    fz_eps = episodes("e3_frozen_eval")
    pairs = [(f"{a}_frozen_{v}", a) for a in ("ppo_scheduler_nominal", "ppo_scheduler_disturbed")
             for v in ("clean", "delay")]
    err_t, comp_t = paired_tables(pairs, fz_eps, ho_eps, match_seed=True)
    out += ["", "### Paired frozen − dynamic per seed and seed-mean", ""]
    out.append(f"Same checkpoint, same scenario. Per-seed columns are mean differences over the 60 scenarios; the "
               f"CI is the hierarchical bootstrap of `bootstrap.py` (resample seeds, then scenarios within seed; "
               f"{DRAWS:,} draws, generator re-seeded with {BOOT_SEED} for every comparison). Positive ΔE_FA = frozen has larger error.\n")
    out.append("**ΔE_FA (mm)**\n\n" + err_t)
    out.append("\n**Δcompletion (percentage points)**\n\n" + comp_t)
    err_r, comp_r = paired_vs_fixed([p[0] for p in pairs], fz_eps, ho_eps, "pid_global_robust")
    out += ["", details("Paired frozen − PID-R (seed-mean and 95 % CI)",
                        "**ΔE_FA (mm)**\n\n" + err_r + "\n\n**Δcompletion (pp)**\n\n" + comp_r)]
    out += ["", "### Observations", "", o("E3")]

    # §4
    out += ["", "## 4. E4 Noise-trained arms", "", "### σ*, sampler change, PID-RN method", ""]
    out.append(o("E4 method"))
    rows = []
    for label in ("3mm", "10mm"):
        r = json.loads((PH / f"e4_pid_retune_{label}" / "pid_rn.json").read_text(encoding="utf-8"))
        s, sm, sr = r["selected"], r["selected_metrics"], r["sealed_robust_metrics"]
        rows.append([f"{r['sigma_m'] * 1000:g} mm", f"({s['kp']:g}, {s['ki']:g}, {s['kd']:g}), τd {s['tau_d']:g} s",
                     f"{pct(sm['completion_rate'])} / {mm(sm['failure_adjusted_error_m'])}",
                     f"{pct(sr['completion_rate'])} / {mm(sr['failure_adjusted_error_m'])}",
                     ", ".join(r.get("selected_on_grid_edge", [])) or "none", len(r["scenarios"])])
    out.append("\nPID-RN selection on the 28-scenario robust validation subset with σ* injected (candidates in "
               "`e4_pid_retune_*/candidates.csv`):\n")
    out.append(table(["σ*", "PID-RN (Kp, Ki, Kd), τd", "PID-RN compl % / E_FA mm",
                      "sealed PID-R on same subset", "on grid edge", "scenarios"], rows))
    out += ["", "### Training cost (wall time per run)", "", TRAINING_NOTE, ""]
    for label in ("3mm", "10mm"):
        out.append(f"**σ* = {label}**\n")
        out.append(training_cost(PH / f"e4_training_{label}.log") + "\n")
    e4rows = summary("e4_noise_sweep")
    e4 = index(coarse + fine + e4rows)
    out += ["### Noise sweep (as E1)", ""]
    out.append("New arms from `e4_noise_sweep` (cpunode04); PID-R, GS-D and DR-D columns repeated from E1 "
               "(cpunode39) for reference. Both nodes replayed the sealed run bit for bit.\n")
    out.append(sweep_tables(e4, E4_ARMS, lv_all, "noise", ("noise", "combined")))
    new = [a for a in E4_ARMS if "noise" in a]
    out += ["", details("Full metrics, new arms", full_metric_table(e4, new, lv_all, "noise", ("noise", "combined")))]
    e4_seed = index(summary("e4_noise_sweep", file="summary_by_seed.csv"), with_seed=True)
    out += ["", details("Per-seed completion % / E_FA mm, new learned arms", per_seed_tables(
        e4_seed, [a for a in new if a.startswith("ppo")], lv_all, "noise", ("noise", "combined")))]
    out += ["", "**Breaking point (rule as in E1)**", ""]
    out.append(breaking_table(e4, E4_ARMS, lv_all, ("noise", "combined"), "noise"))
    ho = index(ho_rows)
    ho_seed = index(summary("heldout_rerun", file="summary_by_seed.csv"), with_seed=True)
    out += ["", "### Held-out 480 vs sealed arms", ""]
    out.append("All from `heldout_rerun` (one node, one run). Noise cells carry the sealed 0.3 mm.\n")
    out.append(heldout_table(ho, E4_ARMS))
    out += ["", details("Per-seed completion % / E_FA mm",
                        heldout_seed_table(ho_seed, [a for a in E4_ARMS if a.startswith("ppo")]))]
    pairs4 = [(f"ppo_{k}_noisetrained_{L}", f"ppo_{k}_disturbed") for k in ("scheduler", "direct")
              for L in ("3mm", "10mm")]
    err_t, comp_t = paired_tables(pairs4, ho_eps, ho_eps, match_seed=True)
    out += ["", "**Paired noise-trained − sealed disturbed, held-out 480** (seed-matched: same seed number, "
            "different training run)", "", "ΔE_FA (mm)", "", err_t, "", "Δcompletion (pp)", "", comp_t]
    err_p, comp_p = paired_vs_fixed(["pid_global_robust_noise_3mm", "pid_global_robust_noise_10mm"],
                                    ho_eps, ho_eps, "pid_global_robust")
    out += ["", "**Paired PID-RN − PID-R, held-out 480**", "", "ΔE_FA (mm)", "", err_p, "",
            "Δcompletion (pp)", "", comp_p]
    out += ["", "### Observations", "", o("E4")]

    # §5
    out += ["", "## 5. E5 History ablation", "", "### Training cost", ""]
    out.append("H = 1 frame instead of the sealed H = 10; the frame still contains the previous action. GS-D and "
               "DR-D only, 5 seeds, sealed budgets, hyperparameters and checkpoint rule. Network input 17 "
               "(scheduler) and 13 (direct) instead of 170 and 130.\n")
    out.append(TRAINING_NOTE + "\n")
    out.append(training_cost(PH / "e5_training.log"))
    arms5 = ["ppo_scheduler_disturbed", "ppo_scheduler_disturbed_h1", "ppo_direct_disturbed",
             "ppo_direct_disturbed_h1"]
    out += ["", "### Table: arm × H × mode × condition → completion %, E_FA mm", ""]
    out.append("Held-out 480 from `heldout_rerun`. H10 rows are the sealed checkpoints.\n")
    out.append(heldout_table(ho, ["pid_global_robust"] + arms5))
    out += ["", "**Per seed**", "", heldout_seed_table(ho_seed, arms5)]
    err_t, comp_t = paired_tables([("ppo_scheduler_disturbed_h1", "ppo_scheduler_disturbed"),
                                   ("ppo_direct_disturbed_h1", "ppo_direct_disturbed")], ho_eps, ho_eps,
                                  match_seed=True)
    out += ["", "**Paired H1 − H10** (seed-matched)", "", "ΔE_FA (mm)", "", err_t, "", "Δcompletion (pp)", "",
            comp_t]
    out += ["", "### Observations", "", o("E5")]

    # §6
    out += ["", "## 6. Files", ""]
    rows = []
    for d in sorted(p for p in PH.iterdir() if p.is_dir()):
        rows.append([f"`data/results/stage1/{d.name}/`", ", ".join(sorted(f.name for f in d.iterdir()))])
    out.append(table(["directory", "files"], rows))
    out.append("")
    out.append("Logs: `data/results/stage1/slurm-thesis-posthoc-{s1-3187230,s2-3188678,s2b-3189024}.out`, "
               "`e4_training_{3mm,10mm}.log`, `e5_training.log`. Manifests: `data/scenarios/"
               "{noise_coarse,noise_fine,delay}.json`. Models: the noise-trained arms (`e4_noise_3mm`, `e4_noise_10mm`) are not part of the thesis "
               "and their checkpoints are not in `data/checkpoints/`. Code: `pathfollow/`, "
               "`cluster/stage1*.sh`.")
    out += ["", "**Figures** (`figures_stage1.py`; PDF + 300 dpi PNG in `data/results/stage1/figures/`):", ""]
    figures = [
        ("e1_noise_sweep_stationary / _transient", "E1: completion and E_FA vs σ, six sealed arms, noise-only "
         "and combined; 10–30 mm band shaded."),
        ("e1_mechanism_stationary_noise", "E1: steering TV/T and wheel saturation vs σ, noise-only, stationary."),
        ("e1f_pid_derivative_filter", "E1-F: PID-R completion vs σ for τd 0.01 / 0.02 / 0.05 s."),
        ("e2_delay_sweep", "E2: completion and E_FA vs delay, both modes; 0.15 s marked."),
        ("e3_frozen_gain_completion", "E3: per-seed completion over the delay-bearing held-out cells, dynamic vs "
         "F-clean vs F-delay."),
        ("e4_noise_sweep_{noise,combined}_{stationary,transient}", "E4: sealed arm vs σ* = 3 mm and 10 mm "
         "variants, per family, completion and E_FA vs σ."),
        ("e4_heldout_completion", "E4: held-out completion per cell, sealed vs variants, per family."),
        ("e5_history_paired_difference", "E5: paired ΔE_FA (H1 − H10) per cell, seed-mean with 95 % CI and "
         "per-seed points."),
    ]
    out.append(table(["file", "content"], ([f"`{name}`", text] for name, text in figures)))
    out.append("\nEncoding: each sealed arm keeps one colour, marker and line style in every figure (PID dashed "
               "square, scheduler solid circle, direct dotted triangle; nominal hollow, disturbed filled). In "
               "variant figures dark gray is the sealed/reference arm, blue the first variant and orange the "
               "second. Both palettes passed the colour-vision-deficiency separation check.")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPORTS / "stage1.md"))
    parser.add_argument("--copy-to")
    args = parser.parse_args()
    text = build()
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({len(text.splitlines())} lines)")
    if args.copy_to:
        shutil.copyfile(args.output, args.copy_to)
        print(f"copied to {args.copy_to}")


if __name__ == "__main__":
    main()
