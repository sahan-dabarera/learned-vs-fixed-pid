"""Build POSTHOC_RESULTS_2.md (experiment plan 2) from artifacts/results/posthoc2.

Every number is read from the result directories. Free-text observations come
from posthoc_observations2.md (sections "## 0 Deviations", "## F0" ... "## F6").

PID-RN rows always come from the ``pidrn_*`` directories (final retune,
Kp 2.0 / Ki 0.5 / Kd 0.5 / tau_d 0.01). The PID-RN rows inside the job-3225586
directories belong to the superseded Kp 4.0 grid-edge pick and are dropped.

    python posthoc_report2.py
    python posthoc_report2.py --copy-to D:/Msc/Thesis_Final/Experiments/POSTHOC_RESULTS_2.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import posthoc_report as r1
from posthoc_report import (
    CONDITIONS, MODES, SEEDS, breaking_table, cell, details, estimate, finished, full_metric_table, index,
    levels_of, mm, paired, pct, per_seed_tables, sweep_tables, table, training_cost,
)
from metrics import read_csv

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "artifacts" / "results" / "posthoc2"
PH = ROOT / "artifacts" / "results" / "posthoc"
MAN = ROOT / "manifests" / "posthoc"
FIG = P2 / "figures"
PIDRN = "pid_global_robust_noise"

SHORT2 = {
    "pid_global_nominal": "PID-N",
    "pid_global_robust": "PID-R",
    PIDRN: "PID-RN",
    "ppo_scheduler_nominal_h1": "GS-N",
    "ppo_scheduler_ranged_h1": "GS-D",
    "ppo_direct_nominal_h1": "DR-N",
    "ppo_direct_ranged_h1": "DR-D",
    "ppo_scheduler_nominal": "GS-N H10",
    "ppo_direct_nominal": "DR-N H10",
    "ppo_scheduler_ranged_h10": "GS-D H10",
    "ppo_direct_ranged_h10": "DR-D H10",
    "ppo_scheduler_disturbed": "GS-D old-sampler H10",
    "ppo_direct_disturbed": "DR-D old-sampler H10",
    "ppo_scheduler_disturbed_h1": "GS-D old-sampler H1",
    "ppo_direct_disturbed_h1": "DR-D old-sampler H1",
    "ppo_scheduler_nominal_h1_frozen_clean": "GS-N F-clean",
    "ppo_scheduler_nominal_h1_frozen_delay": "GS-N F-delay",
    "ppo_scheduler_ranged_h1_frozen_clean": "GS-D F-clean",
    "ppo_scheduler_ranged_h1_frozen_delay": "GS-D F-delay",
}
r1.SHORT.update(SHORT2)  # the shared table helpers read names from r1.SHORT
SHORT = r1.SHORT

MAIN = ["pid_global_nominal", "pid_global_robust", PIDRN, "ppo_scheduler_nominal_h1", "ppo_scheduler_ranged_h1",
        "ppo_direct_nominal_h1", "ppo_direct_ranged_h1"]
LEARNED = ["ppo_scheduler_nominal_h1", "ppo_scheduler_ranged_h1", "ppo_direct_nominal_h1", "ppo_direct_ranged_h1"]
H10 = {"ppo_scheduler_nominal_h1": "ppo_scheduler_nominal", "ppo_direct_nominal_h1": "ppo_direct_nominal",
       "ppo_scheduler_ranged_h1": "ppo_scheduler_ranged_h10", "ppo_direct_ranged_h1": "ppo_direct_ranged_h10"}
ABLATION = list(H10.values())
OLD = ["ppo_scheduler_disturbed", "ppo_direct_disturbed", "ppo_scheduler_disturbed_h1", "ppo_direct_disturbed_h1"]
ALL_ARMS = MAIN + ABLATION + OLD

TRAINING_DELAY = """\
**Training disturbance sampler as implemented** (`env.py::_sample_disturbance`, `config.py`).

- **Condition per episode** (both disturbed samplers): nominal 25 %, delay 25 %, noise 15 %, combined 35 %.
- **Timing:** each disturbed episode is stationary (disturbance from t = 0) with p = 0.5. Otherwise it is
  transient, with 1, 2 or 3 changes (p = 0.50 / 0.30 / 0.20) at times uniform in
  [1 s, max(1 s, 0.8 · ideal duration)]. Each change after the first returns the channel to zero with p = 0.35;
  otherwise it draws a new value. In combined episodes delay and noise change at the same times.
- **Delay value per draw:**
  - old sampler (sealed GS-D/DR-D, E5 H1): 0.15 s with p = 0.5, otherwise uniform [0, 0.15] s.
    Mean 0.1125 s, maximum 0.15 s;
  - ranged sampler (plan 2 GS-D, DR-D, GS-D H10, DR-D H10): uniform [0, 0.3] s, no point mass. Mean 0.15 s.
- **Noise value per draw:**
  - old sampler: 0.3 mm with p = 0.5, otherwise uniform [0, 0.4] mm;
  - ranged sampler: log-uniform [0.1, 10] mm.
- **Quantisation:** delays are applied in whole physics steps (`round(delay / DT)`).
- **Nominal arms** (GS-N, DR-N, both H) never see a disturbance in training.

Levels of the F3 sweep inside the training range:
- ranged arms: 0–0.3 s (levels 0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3);
- old-sampler arms: 0–0.15 s;
- nominal arms: 0 only. PID gains are fixed.
"""

DYNAMIC = """\
**Schedule** (`posthoc_manifests.py dynamic`, seed 20261116, identical for every arm).

- **Base scenarios:** the 60 held-out stationary delay records (20 paths × 3 speeds), noise off.
- **Delay:** the initial value is drawn uniform in [0, D_max] at t = 0. Each value is held for a time drawn
  uniform in [1 s, 3 s]; the next value is drawn uniform in [0, D_max]. This repeats up to the episode time limit
  (2 × ideal duration + 5 s).
- **Ranges:** D_max = 0.15 s (the old training range), 0.3 s (the ranged training range) and 0.45 s.
- **Combined variant:** the same delay schedule, plus noise on its own schedule. Holds are drawn uniform in
  [1 s, 3 s]; values log-uniform in [0.1, 10] mm.
- **Delay buffer** (unchanged, `env.py::_set_delay_steps`):
  - when the delay grows, the queue is padded with copies of its oldest command, or with zero commands if
    the queue is empty;
  - when it shrinks, the oldest commands are dropped;
  - at t = 0 every episode with a non-zero initial delay starts with that many zero commands queued. This is
    the same as in the sealed run.

**How it differs from training.** The ranged training sampler changes the delay 1–3 times in half of its
delay-bearing episodes, at times spread over the first 80 % of the ideal duration, and with returns to zero.
F4 changes it every 1–3 s through the whole episode (counts below), with no forced return to zero.
"""


# ---------------------------------------------------------------- loading

def _rows(directory: Path, file: str) -> list[dict[str, str]]:
    return read_csv(directory / file)


def load(main_dirs: list[str], pidrn_dir: str | None, file: str = "summary.csv",
         old_dirs: tuple[str, ...] = ()) -> list[dict[str, str]]:
    """Rows from plan-2 directories with PID-RN taken from its final directory."""
    rows: list[dict[str, str]] = []
    for name in main_dirs:
        rows += [row for row in _rows(P2 / name, file) if row["controller"] != PIDRN]
    if pidrn_dir:
        rows += _rows(P2 / pidrn_dir, file)
    for name in old_dirs:
        rows += [row for row in _rows(PH / name, file)
                 if row["controller"] in OLD_DIR_ARMS]
    return rows


OLD_DIR_ARMS = ("pid_global_nominal", "pid_global_robust", "ppo_scheduler_nominal", "ppo_direct_nominal")


def meta(directory: Path, file: str = "metadata.json") -> dict[str, Any]:
    return json.loads((directory / file).read_text(encoding="utf-8"))


def bookkeeping(name: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((MAN / f"{name}.json").read_text(encoding="utf-8"))
    return payload["posthoc"]["scenarios"]


# ---------------------------------------------------------------- paired helpers

def pair_rows(rows_a, rows_b, a, b, mode, condition, match_seed, level=None):
    if level is not None:
        keep = lambda row: row.get("level") and abs(float(row["level"]) - level) < 1e-9  # noqa: E731
        rows_a = [row for row in rows_a if keep(row)]
        rows_b = [row for row in rows_b if keep(row)]
    return paired(rows_a, rows_b, a, b, mode, condition, match_seed)


def paired_block(pairs, rows, cells, level=None) -> tuple[str, str]:
    """pairs: (a, b, match_seed). Returns (E_FA table, completion table) with per-seed columns."""
    err_rows, comp_rows = [], []
    for a, b, match_seed in pairs:
        for mode, condition in cells:
            err, comp = pair_rows(rows, rows, a, b, mode, condition, match_seed, level)
            if not err:
                continue
            e, c = estimate(err), estimate(comp)
            label = [f"{SHORT[a]} − {SHORT[b]}", mode, condition]
            err_rows.append(label + [f"{np.mean(err[s]) * 1000:+.2f}" if s in err else "–" for s in SEEDS]
                            + [f"**{e[0] * 1000:+.2f}**", f"[{e[1] * 1000:+.2f}, {e[2] * 1000:+.2f}]"])
            comp_rows.append(label + [f"{np.mean(comp[s]) * 100:+.1f}" if s in comp else "–" for s in SEEDS]
                             + [f"**{c[0] * 100:+.1f}**", f"[{c[1] * 100:+.1f}, {c[2] * 100:+.1f}]"])
    header = ["pair", "mode", "condition"] + [f"s{s}" for s in SEEDS] + ["seed-mean", "95 % CI"]
    return table(header, err_rows), table(header, comp_rows)


HELDOUT_CELLS = [(m, c) for m in MODES for c in CONDITIONS]


def comparison_pairs() -> list[tuple[str, str, bool]]:
    pairs = []
    for arm in LEARNED:
        pairs += [(arm, "pid_global_robust", False), (arm, PIDRN, False)]
    pairs += [("ppo_scheduler_ranged_h1", "ppo_direct_ranged_h1", True),
              ("ppo_scheduler_nominal_h1", "ppo_direct_nominal_h1", True),
              ("ppo_scheduler_ranged_h1", "ppo_scheduler_nominal_h1", True),
              ("ppo_direct_ranged_h1", "ppo_direct_nominal_h1", True)]
    return pairs


def history_pairs() -> list[tuple[str, str, bool]]:
    return [(arm, H10[arm], True) for arm in LEARNED]


# ---------------------------------------------------------------- sections

def run_record() -> str:
    outputs = [
        ("PID-RN retune round 1 (Kp 4–40, superseded)", "pid_retune", "pid_rn.json"),
        ("F3 existing arms", "f3_delay_existing", "metadata.json"),
        ("F4 existing arms", "f4_dynamic_existing", "metadata.json"),
        ("F2 ranged held-out", "f2_heldout_ranged", "metadata.json"),
        ("F2 fixed held-out", "f2_heldout_fixed", "metadata.json"),
        ("F3 new arms", "f3_delay_new", "metadata.json"),
        ("F4 new arms", "f4_dynamic_new", "metadata.json"),
        ("F6 frozen gains (validation)", "f6_frozen_gains", "frozen_gains.json"),
        ("F6 frozen on ranged held-out", "f6_frozen_eval", "metadata.json"),
        ("F5 noise sweeps", "f5_noise_sweep", "metadata.json"),
        ("PID-RN retune round 2 (Kp 1–40)", "pid_retune_wide", "pid_rn.json"),
        ("PID-RN on ranged held-out", "pidrn_f2_heldout_ranged", "metadata.json"),
        ("PID-RN on fixed held-out", "pidrn_f2_heldout_fixed", "metadata.json"),
        ("PID-RN on F3", "pidrn_f3_delay", "metadata.json"),
        ("PID-RN on F4", "pidrn_f4_dynamic", "metadata.json"),
        ("PID-RN on F5", "pidrn_f5_noise_sweep", "metadata.json"),
        ("PID-RN retune round 3 (fine)", "pid_retune_wide_fine", "pid_rn.json"),
    ]
    rows = []
    for label, name, file in outputs:
        m = meta(P2 / name, file)
        machine = m["machine"]
        rows.append([label, f"`{name}`", machine.get("slurm_job_id", "?"), machine.get("node", "?").split(".")[0],
                     f"`{m['git']['commit'][:7]}`" + (" dirty" if m["git"].get("dirty") else ""),
                     m.get("episodes", "–"), f"{m['wall_time_s'] / 60:.1f}"])
    record = table(["step", "output", "SLURM job", "node", "commit", "episodes", "wall min"], rows)
    m = meta(P2 / "f2_heldout_ranged")
    versions = ", ".join(f"{k} {v}" for k, v in sorted(m.get("packages", {}).items()))
    checks = []
    for name in ("f2_heldout_fixed", "f3_delay_existing"):
        c = meta(P2 / name, "sealed_check.json")
        checks.append([f"`{name}`", c["rows_checked"], c["bit_identical_rows"], len(c["completion_mismatches"]),
                       f"{c['max_abs_diff_finished_rows']['failure_adjusted_error_m'] * 1000:.4f}"])
    fine_same = all(
        _rows(P2 / f"pidrn_{s}", "episodes.csv") and
        [_strip(x) for x in _rows(P2 / f"pidrn_{s}", "episodes.csv")] ==
        [_strip(x) for x in _rows(P2 / f"pidrn_{s}_fine", "episodes.csv")]
        for s in ("f2_heldout_ranged", "f2_heldout_fixed", "f3_delay", "f4_dynamic", "f5_noise_sweep")
    )
    return "\n\n".join([
        record,
        f"- **Machine:** {m['machine']['processor']}, {m['machine']['platform']}, Python {m['machine']['python']}.",
        f"- **Packages:** {versions}." if versions else "",
        "- **Parity gate:** `check_parity.py` passed at the start of all three jobs "
        "(`slurm-thesis-plan2-3225586.out` and the two `slurm-thesis-plan2-pidrn-*.out` logs).",
        "- **Replay against the sealed run:**\n\n" + table(
            ["output", "rows checked", "bit-identical", "completion mismatches", "max |ΔE_FA| mm"], checks),
        f"- **PID-RN rounds 2 and 3** give identical rows on all five manifests (timings excluded): "
        f"{'yes' if fine_same else 'NO'}.",
    ])


def _strip(row: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in row.items() if "time" not in k and "inference" not in k}


def pid_retune_section() -> str:
    rows = []
    for label, name in (("1 (Kp 4–40, Kd 0.5–8, τd 0.01–0.1)", "pid_retune"),
                        ("2 (Kp 1–40, Kd 0.5–8, τd 0.01–0.1)", "pid_retune_wide"),
                        ("3 (Kp 1.5–3, Kd 0–1, τd 0.005–0.02)", "pid_retune_wide_fine")):
        m = meta(P2 / name, "pid_rn.json")
        s, x = m["selected"], m["selected_metrics"]
        rows.append([label, len(read_csv(P2 / name / "candidates.csv")),
                     f"({s['kp']:g}, {s['ki']:g}, {s['kd']:g}, τd {s['tau_d']:g})",
                     ", ".join(m.get("selected_on_grid_edge", [])) or "none",
                     pct(x["completion_rate"]), mm(x["failure_adjusted_error_m"]),
                     f"{x['steer_total_variation_per_s']:.1f}"])
    sealed = meta(P2 / "pid_retune_wide", "pid_rn.json")["sealed_robust_metrics"]
    return (table(["round (grid)", "candidates", "selected (Kp, Ki, Kd)", "on grid edge", "val compl %",
                   "val E_FA mm", "steer TV/T"], rows)
            + f"\n\n- **Validation subset:** the 28-scenario robust subset of `validation_ranged.json`. "
              f"Selection rule: `calibrate_pid.select_practical_candidate`.\n"
            f"- **Sealed PID-R (8, 0.5, 4.4) on the same subset:** {pct(sealed['completion_rate'])} % completion, "
            f"{mm(sealed['failure_adjusted_error_m'])} mm.\n"
            "- **Final PID-RN:** the rule applied to all candidates of the three rounds together selects the "
            "round-2/3 point, which is used everywhere below.")


def f0_section() -> str:
    rows = read_csv(P2 / "f0_validation" / "validation_selected.csv")
    by = defaultdict(dict)
    for row in rows:
        by[row["controller"]][row["seed"]] = row
    held = {
        "validation.json": index(read_csv(P2 / "f2_heldout_fixed" / "summary.csv")),
        "validation_ranged.json": index(load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged")),
    }
    order = ["ppo_scheduler_nominal_h1", "ppo_scheduler_nominal", "ppo_direct_nominal_h1", "ppo_direct_nominal",
             "ppo_scheduler_ranged_h1", "ppo_scheduler_ranged_h10", "ppo_direct_ranged_h1", "ppo_direct_ranged_h10",
             "ppo_scheduler_disturbed_h1", "ppo_scheduler_disturbed", "ppo_direct_disturbed_h1",
             "ppo_direct_disturbed"]
    out = []
    for arm in order:
        seeds = by[arm]
        manifest = next(iter(seeds.values()))["validation_manifest"]
        comp = np.mean([float(seeds[s]["completion_rate"]) for s in SEEDS])
        err = np.mean([float(seeds[s]["failure_adjusted_error_m"]) for s in SEEDS])
        idx = held[manifest]
        cells = [idx.get((arm, None, "stationary", "nominal")), idx.get((arm, None, "transient", "combined"))]
        h_comp = np.mean([float(c["completion_rate"]) for c in cells])
        h_err = np.mean([float(c["failure_adjusted_error_m"]) for c in cells])
        out.append([SHORT[arm], f"`{manifest}`"]
                   + [f"{pct(seeds[s]['completion_rate'])} / {mm(seeds[s]['failure_adjusted_error_m'])}" for s in SEEDS]
                   + [f"**{pct(comp)} / {mm(err)}**", f"{pct(h_comp)} / {mm(h_err)}"])
    return (table(["arm", "validation set", *[f"s{s}" for s in SEEDS], "val seed-mean",
                   "held-out, same cells"], out)
            + "\n\n- **Validation:** completion % / E_FA mm of the checkpoint each run kept. Checkpoint scenarios are "
              "the 24 validation scenarios that are stationary-nominal or transient-combined.\n"
              "- **Held-out, same cells:** the mean of the held-out stationary-nominal and transient-combined cells. "
              "Ranged arms use the ranged held-out set; arms selected on `validation.json` use the fixed held-out set.")


def heldout_metric_table(idx, arms) -> str:
    rows = []
    for arm in arms:
        for mode in MODES:
            for condition in CONDITIONS:
                row = idx.get((arm, None, mode, condition))
                if row is None:
                    continue
                rows.append([SHORT[arm], mode, condition, pct(row["completion_rate"]),
                             mm(row["failure_adjusted_error_m"]), mm(row["mean_distance_m"]),
                             f"{float(row['steer_total_variation_per_s']):.1f}",
                             pct(row["wheel_saturation_fraction"]), f"{float(row['speed_ratio']):.3f}"])
    return table(["arm", "mode", "condition", "compl %", "E_FA mm", "path dist mm", "steer TV/T (1/s)",
                  "wheel sat %", "v̄/v_ref"], rows)


def f2_section() -> str:
    rows = load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged")
    by_seed = load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged", file="summary_by_seed.csv")
    ep = load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged", file="episodes.csv")
    idx, idx_seed = index(rows), index(by_seed, with_seed=True)
    fixed = load(["f2_heldout_fixed"], "pidrn_f2_heldout_fixed")
    book = bookkeeping("held_out_ranged")
    d = np.array([b["delay_s"] for b in book.values() if b["delay_s"] > 0])
    n = np.array([b["noise_m"] for b in book.values() if b["noise_m"] > 0]) * 1000
    draws = (f"Ranged held-out draws (seed 20261116): {len(d)} delay-bearing records, delay min {d.min():.3f} / "
             f"median {np.median(d):.3f} / max {d.max():.3f} s; {len(n)} noise-bearing records, noise min "
             f"{n.min():.2f} / median {np.median(n):.2f} / max {n.max():.2f} mm.")
    metric = heldout_metric_table(idx, MAIN + ABLATION)
    cmp_err, cmp_comp = paired_block(comparison_pairs(), ep, HELDOUT_CELLS)
    h_err, h_comp = paired_block(history_pairs(), ep, HELDOUT_CELLS)
    return "\n\n".join([
        draws,
        "### 3.1 Main table, ranged held-out (completion % / E_FA mm, seed-pooled)",
        r1.heldout_table(idx, MAIN + ABLATION + OLD),
        details("All metrics (main arms and H10 ablation)", metric),
        details("Per seed", r1.heldout_seed_table(idx_seed, LEARNED + ABLATION + OLD)),
        "### 3.2 Paired differences, ranged held-out (seed-mean, hierarchical bootstrap, 10,000 replicates)",
        "Learned − PID: arm seeds against the single PID instance. Learned − learned: matched by training seed.",
        "**ΔE_FA (mm)**", cmp_err, "**ΔCompletion (percentage points)**", cmp_comp,
        "### 3.3 History: H = 1 − H = 10 (matched seeds)",
        "**ΔE_FA (mm)**", h_err, "**ΔCompletion (pp)**", h_comp,
        "### 3.4 Fixed-severity held-out (sealed 0.15 s / 0.3 mm), for comparison",
        r1.heldout_table(index(fixed), MAIN + ABLATION + OLD),
    ])


def f3_section() -> str:
    rows = load(["f3_delay_existing", "f3_delay_new"], "pidrn_f3_delay")
    by_seed = load(["f3_delay_existing", "f3_delay_new"], "pidrn_f3_delay", file="summary_by_seed.csv")
    idx, idx_seed, levels = index(rows), index(by_seed, with_seed=True), levels_of(rows)
    arms = MAIN + ABLATION + OLD
    return "\n\n".join([
        TRAINING_DELAY,
        "Delay only, noise off. Stationary: delay from t = 0. Transient: step at 6 s. 60 scenarios per mode and level. "
        "Every level was run for every arm; no early stop.",
        sweep_tables(idx, arms, levels, "delay", ("delay",)),
        "**Breaking points** (E1 rule: first level with completion < 100 % or E_FA > 2 × the 0 s value; "
        "modes pooled or separate)",
        breaking_table(idx, arms, levels, ("delay",), "delay"),
        details("All metrics", full_metric_table(idx, MAIN + ABLATION, levels, "delay", ("delay",))),
        details("Per seed (learned arms)", per_seed_tables(idx_seed, LEARNED + ABLATION, levels, "delay", ("delay",))),
    ])


def zero_command_table() -> str:
    from core.dynamics import DT

    book = bookkeeping("dynamic")
    ep = load(["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic", file="episodes.csv")
    durations = defaultdict(list)
    for row in ep:
        durations[row["scenario_id"]].append(float(row["duration_s"]))
    rows = []
    for level in (0.15, 0.3, 0.45):
        items = [(sid, b) for sid, b in book.items() if abs(b["level"] - level) < 1e-9 and b["variant"] == "delay"]
        changes, zero_start, from_empty, from_empty_run = [], 0, 0, 0
        for sid, b in items:
            sched = b["delay_schedule"]
            steps = [int(round(v / DT)) for _, v in sched]
            changes.append(len(sched) - 1)
            zero_start += steps[0] == 0
            hits = [t for (t, _), a, c in zip(sched[1:], steps, steps[1:]) if a == 0 and c > 0]
            from_empty += len(hits)
            run_end = float(np.median(durations[sid] + durations[sid[: -len("-delay")] + "-combined"]))
            from_empty_run += sum(1 for t in hits if t < run_end)
        rows.append([f"{level:g} s", f"{np.mean(changes):.1f} ({min(changes)}–{max(changes)})",
                     zero_start, from_empty, from_empty_run])
    return table(["D_max", "changes per scenario to the time limit, mean (range)",
                  "scenarios starting at 0 steps", "0-step → >0 changes (to time limit)",
                  "… before the median episode end"], rows)


def f4_section() -> str:
    rows = load(["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic")
    ep = load(["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic", file="episodes.csv")
    by_seed = load(["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic", file="summary_by_seed.csv")
    idx, idx_seed, levels = index(rows), index(by_seed, with_seed=True), levels_of(rows)
    arms = MAIN + ABLATION + OLD
    main = []
    for variant in ("delay", "combined"):
        body = table(["D_max"] + [SHORT[a] for a in arms],
                     ([f"{lv:g} s"] + [cell(idx.get((a, lv, "stationary", variant))) for a in arms] for lv in levels))
        main.append(f"**{variant}** (completion % / E_FA mm, seed-pooled)\n\n{body}")
    pairs = [(a, b, m) for a, b, m in comparison_pairs() if b in ("pid_global_robust", PIDRN)] + history_pairs()
    blocks = []
    for lv in levels:
        e, c = paired_block(pairs, ep, [("stationary", "delay"), ("stationary", "combined")], level=lv)
        blocks.append(f"**D_max = {lv:g} s, ΔE_FA (mm)**\n\n{e}\n\n**D_max = {lv:g} s, ΔCompletion (pp)**\n\n{c}")
    return "\n\n".join([
        DYNAMIC,
        "**Change counts and zero-command intervals**",
        zero_command_table(),
        "- **Change counts** run up to the episode time limit; an episode that finishes early sees fewer.\n"
        "- **\"0-step → >0\" changes** are the cases where the delay grows from an empty queue, so the "
        "first new slots are filled with zero commands.\n"
        "- **The last column** keeps only changes before the median finish time over all arms and seeds on that "
        "scenario.",
        "### 5.1 Results", *main,
        details("All metrics (completion, E_FA, path distance, saturation, …)",
                full_metric_table(idx, MAIN + ABLATION, levels, "delay", ("delay", "combined"))),
        details("Per seed (learned arms)",
                per_seed_tables(idx_seed, LEARNED + ABLATION, levels, "delay", ("delay", "combined"))),
        "### 5.2 Paired differences (learned − PID-R, − PID-RN; H1 − H10)", *blocks,
        "Figure: `figures/f4_dynamic_delay.{pdf,png}`.",
    ])


def f5_section() -> str:
    rows = load(["f5_noise_sweep"], "pidrn_f5_noise_sweep", old_dirs=("e1_noise_coarse", "e1_noise_fine"))
    by_seed = load(["f5_noise_sweep"], "pidrn_f5_noise_sweep", file="summary_by_seed.csv")
    idx, idx_seed, levels = index(rows), index(by_seed, with_seed=True), levels_of(rows)
    arms = MAIN + ABLATION
    return "\n\n".join([
        "Same manifests as E1 (coarse 0.3–100 mm plus 12 fine levels; noise only, and noise + 0.15 s delay; both "
        "modes). PID-N, PID-R, GS-N H10 and DR-N H10 rows are the E1 rows: same scenarios and controller instances, and "
        "the sealed replay checks show that rows reproduce bit for bit on this CPU model.",
        sweep_tables(idx, arms, levels, "noise", ("noise", "combined")),
        "**Breaking points** (E1 rule)",
        breaking_table(idx, arms, levels, ("noise", "combined"), "noise"),
        details("All metrics, incl. mechanism columns (steering TV/T, wheel saturation)",
                full_metric_table(idx, arms, levels, "noise", ("noise", "combined"))),
        details("Per seed (learned arms)", per_seed_tables(idx_seed, LEARNED + ABLATION, levels, "noise",
                                                           ("noise", "combined"))),
    ])


DELAY_CELLS = [("stationary", "delay"), ("stationary", "combined"), ("transient", "delay"), ("transient", "combined")]


def seed_completion(rows, controller, cells):
    by_seed = defaultdict(list)
    for row in rows:
        if row["controller"] == controller and (row["evaluation_mode"], row["condition"]) in cells:
            by_seed[row["training_seed"]].append(finished(row))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def f6_section() -> str:
    gains = meta(P2 / "f6_frozen_gains", "frozen_gains.json")["gains"]
    g_rows = []
    for arm in ("ppo_scheduler_nominal_h1", "ppo_scheduler_ranged_h1"):
        for s in SEEDS:
            v = gains[arm][s]
            g_rows.append([SHORT[arm], s] + [
                f"{v[k]['mean']['kp']:.2f} / {v[k]['mean']['ki']:.3f} / {v[k]['mean']['kd']:.2f}" for k in ("clean", "delay")])
    ep = read_csv(P2 / "f6_frozen_eval" / "episodes.csv")
    rows = read_csv(P2 / "f6_frozen_eval" / "summary_by_seed.csv")
    idx_seed = index(rows, with_seed=True)
    c_rows = []
    for arm in ("ppo_scheduler_nominal_h1", "ppo_scheduler_ranged_h1"):
        per = {v: seed_completion(ep, name, DELAY_CELLS) for v, name in
               (("dyn", arm), ("clean", f"{arm}_frozen_clean"), ("delay", f"{arm}_frozen_delay"))}
        for s in SEEDS + ("mean",):
            vals = {v: (np.mean(list(p.values())) if s == "mean" else p[s]) for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            flag = "**yes**" if s != "mean" and gap > 0.05 else ""
            c_rows.append([SHORT[arm], s, pct(vals["dyn"]), pct(vals["clean"]), pct(vals["delay"]),
                           f"{gap * 100:+.1f}", flag])
    arms = ["ppo_scheduler_nominal_h1", "ppo_scheduler_nominal_h1_frozen_clean", "ppo_scheduler_nominal_h1_frozen_delay",
            "ppo_scheduler_ranged_h1", "ppo_scheduler_ranged_h1_frozen_clean", "ppo_scheduler_ranged_h1_frozen_delay"]
    return "\n\n".join([
        "Frozen gains: the time-averaged applied (Kp, Ki, Kd) of each H = 1 scheduler checkpoint on the sealed "
        "validation manifest (F-clean: nominal scenarios; F-delay: delay and combined scenarios; mean). "
        "They are evaluated as constant gains on the ranged held-out 480.",
        table(["arm", "seed", "F-clean Kp / Ki / Kd", "F-delay Kp / Ki / Kd"], g_rows),
        "**Completion in the four delay-bearing cells** (stationary/transient × delay/combined, %)",
        table(["arm", "seed", "dynamic", "F-clean", "F-delay", "dynamic − best frozen (pp)", "flag (> 5 pp)"], c_rows),
        details("All cells per seed", r1.heldout_seed_table(idx_seed, arms)),
    ])


def observations() -> dict[str, str]:
    path = ROOT / "posthoc_observations2.md"
    if not path.exists():
        return {}
    sections, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return {k: v.strip() for k, v in sections.items()}


FIGURES = [
    ("f2_heldout_main", "F2: completion and E_FA of the seven main arms in every ranged held-out cell."),
    ("f2_history_paired_difference", "F2: paired ΔE_FA, H = 1 − H = 10, per arm and cell (seed-mean, 95 % CI, seeds)."),
    ("f3_delay_sweep", "F3: completion and E_FA vs constant delay 0–0.8 s, main arms; training ranges shaded."),
    ("f4_dynamic_delay", "F4: completion and E_FA vs dynamic-delay range D_max, delay-only and combined."),
    ("f5_noise_sweep_stationary", "F5: completion and E_FA vs noise, stationary, main arms."),
    ("f5_noise_sweep_transient", "F5: completion and E_FA vs noise, transient, main arms."),
    ("f5_mechanism_stationary_noise", "F5: steering TV/T and wheel saturation vs noise (stationary, noise only)."),
    ("f6_frozen_gain_completion", "F6: delay-cell completion per seed, dynamic vs frozen gains (H = 1 schedulers)."),
]


def build() -> str:
    obs = observations()
    o = lambda key: obs.get(key, "_(observations not written yet)_")  # noqa: E731
    names = ("| short name | arm | history | training |\n|---|---|---|---|\n"
             "| PID-N / PID-R | sealed fixed PIDs (250, 4.545, 0.397) / (8, 0.5, 4.4) | – | – |\n"
             "| PID-RN | fixed PID retuned on the ranged validation subset (2.0, 0.5, 0.5, τd 0.01) | – | – |\n"
             "| GS-N / DR-N | scheduler / direct, nominal sampler | H = 1 | new (F1) |\n"
             "| GS-D / DR-D | scheduler / direct, ranged sampler | H = 1 | new (main) |\n"
             "| GS-N H10 / DR-N H10 | sealed nominal checkpoints | H = 10 | sealed |\n"
             "| GS-D H10 / DR-D H10 | ranged sampler | H = 10 | new (history ablation) |\n"
             "| GS-D / DR-D old-sampler H10 | sealed disturbed checkpoints | H = 10 | sealed |\n"
             "| GS-D / DR-D old-sampler H1 | E5 checkpoints (old sampler) | H = 1 | results 1, E5 |")
    out = [
        "# Experiment Results 2",
        "Plan: `Experiment_Plan_2.md`, as revised with the author on 2026-11-15/16 (see §0). Generated by "
        "`posthoc_report2.py` from `artifacts/results/posthoc2/`. All rates are seed-pooled unless marked; "
        "E_FA is the failure-adjusted error in mm.",
        "**Arm names in this document**", names,
        "## 0. Run record",
        run_record(),
        "### 0.1 Deviations from `Experiment_Plan_2.md`", o("0 Deviations"),
        "### 0.2 PID-RN selection", pid_retune_section(),
        "## 1. F0 Validation check", f0_section(), o("F0"),
        "## 2. F1 Training cost",
        "All 30 plan-2 runs trained at once in job 3225586 on cpunode13 (64 cores), beside the PID-RN retune and the "
        "F3/F4 runs of the existing arms (30 workers each).",
        "**Nominal H = 1 (F1)**", training_cost(P2 / "training_nominal_h1.log"),
        "**Ranged H = 1 (main)**", training_cost(P2 / "training_ranged_h1.log"),
        "**Ranged H = 10 (history ablation)**", training_cost(P2 / "training_ranged_h10.log"),
        "## 3. F2 Held-out 480", f2_section(), o("F2"),
        "## 4. F3 Constant-delay sweep, 0–0.8 s", f3_section(), o("F3"),
        "## 5. F4 Dynamic delay", f4_section(), o("F4"),
        "## 6. F5 Noise sweep, H = 1 arms", f5_section(), o("F5"),
        "## 7. F6 Frozen gains, H = 1 schedulers", f6_section(), o("F6"),
        "## 8. F7 Noise-trained H = 1 arms",
        "Merged into the main design. With the author's revision, the main disturbed arms (GS-D, DR-D) are "
        "trained on noise log-uniform in [0.1, 10] mm (together with delay uniform in [0, 0.3] s), and PID-RN is the "
        "noise-aware PID baseline. A separate σ* = 3 mm H = 1 run was not made.",
        "## 9. Observations",
        "Facts per section are written under each section above (F0–F6). No interpretation.",
        "## 10. Files",
        "- **Results:** `artifacts/results/posthoc2/<step>/` (`episodes.csv`, `summary.csv`, `summary_by_seed.csv`, "
        "`metadata.json`).\n- **Models:** `artifacts/models/posthoc/plan2/`.\n- **Manifests:** "
        "`manifests/posthoc/{held_out_ranged,validation_ranged,dynamic,delay_wide}.json`.\n- **Code:** "
        "`posthoc_manifests.py`, `posthoc_validation_check.py`, `posthoc_pid_retune.py`, `posthoc_report2.py`, "
        "`posthoc_figures2.py`.\n- **Jobs:** `slurm_plan2.sh`, `slurm_plan2_pidrn.sh`.",
        "**Figures** (`artifacts/results/posthoc2/figures/`, PDF + 300 dpi PNG). Main-arm colours are fixed across "
        "figures, with a marker and line style per arm (PID square/dashed, PID-RN diamond, scheduler circle/solid, "
        "direct triangle/dotted; nominal hollow, disturbed filled).",
        table(["file", "content"], ([f"`{n}`", c] for n, c in FIGURES)),
    ]
    return "\n\n".join(part for part in out if part) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "POSTHOC_RESULTS_2.md"))
    parser.add_argument("--copy-to")
    args = parser.parse_args()
    text = build()
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({text.count(chr(10))} lines)")
    if args.copy_to:
        shutil.copyfile(args.output, args.copy_to)
        print(f"copied to {args.copy_to}")


if __name__ == "__main__":
    main()
