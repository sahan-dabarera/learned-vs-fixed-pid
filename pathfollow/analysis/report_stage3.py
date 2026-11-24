"""Build stage3.md (experiment stage 3) from data/results/stage3.

Stage-2 rows (PID arms, DR arms, stage-2 schedulers) are reused from
data/results/stage2 through report_stage2.load; job 3281395 re-ran a
slice of them bit-identically (§0). The G3 decision is computed here from the
validation scores of the selected checkpoints, with the rule fixed in
the stage-3 plan; nothing in this file names the winner.

    python -m pathfollow.analysis.report_stage3
    python -m pathfollow.analysis.report_stage3 --copy-to path/to/copy.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pathfollow import config
from pathfollow.analysis import report_stage1 as r1
from pathfollow.analysis import report_stage2 as r2
from pathfollow.analysis.report_stage1 import (
    SEEDS, breaking_table, cell, details, full_metric_table, index, levels_of, mm, pct,
    sweep_tables, table, training_cost,
)
from pathfollow.metrics import read_csv

ROOT = config.ROOT
NOTES = Path(__file__).resolve().parent / "notes"
REPORTS = config.RESULTS / "reports"
P3 = config.RESULTS / "stage3"
M2 = config.CHECKPOINTS
M3 = config.CHECKPOINTS
FIG = P3 / "figures"
PIDRN = r2.PIDRN

GSD, GSDW, GSD2M, GSDW2M = ("ppo_scheduler_ranged_h1", "ppo_scheduler_ranged_h1_wide",
                            "ppo_scheduler_ranged_h1_2m", "ppo_scheduler_ranged_h1_wide_2m")
GSN, GSNW = "ppo_scheduler_nominal_h1", "ppo_scheduler_nominal_h1_wide"
GSDH10, GSDWH10 = "ppo_scheduler_ranged_h10", "ppo_scheduler_ranged_h10_wide"
DRD, DRN = "ppo_direct_ranged_h1", "ppo_direct_nominal_h1"
CANDIDATES = [GSD, GSDW, GSD2M, GSDW2M]

r1.SHORT.update({
    GSDW: "GS-D-W", GSNW: "GS-N-W", GSDWH10: "GS-D-W H10", GSD2M: "GS-D 2M", GSDW2M: "GS-D-W 2M",
    f"{GSDW}_frozen_clean": "GS-D-W F-clean", f"{GSDW}_frozen_delay": "GS-D-W F-delay",
    f"{GSNW}_frozen_clean": "GS-N-W F-clean", f"{GSNW}_frozen_delay": "GS-N-W F-delay",
})
SHORT = r1.SHORT

SETS = {  # stage-2 dirs, stage-2 PID-RN dir, stage-3 dir stem, old E1 dirs
    "heldout_ranged": (["f2_heldout_ranged"], "pidrn_f2_heldout_ranged", "g4_heldout_ranged", ()),
    "heldout_fixed": (["f2_heldout_fixed"], "pidrn_f2_heldout_fixed", "g4_heldout_fixed", ()),
    "delay": (["f3_delay_existing", "f3_delay_new"], "pidrn_f3_delay", "g4_delay", ()),
    "dynamic": (["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic", "g4_dynamic", ()),
    "noise": (["f5_noise_sweep"], "pidrn_f5_noise_sweep", "g4_noise", ("e1_noise_coarse", "e1_noise_fine")),
}


def load(name: str, file: str = "summary.csv") -> list[dict[str, str]]:
    dirs, pidrn, stem, old = SETS[name]
    rows = r2.load(dirs, pidrn, file=file, old_dirs=old if file == "summary.csv" else ())
    for tag in ("g1", "g2"):
        rows += read_csv(P3 / f"{stem}_{tag}" / file)
    return rows


def meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- validation / decision

def validation() -> dict[str, dict[str, dict[str, str]]]:
    by = defaultdict(dict)
    for row in read_csv(P3 / "g3_validation" / "validation_selected.csv"):
        by[row["controller"]][row["seed"]] = row
    return by


def seed_mean(rows: dict[str, dict[str, str]]) -> tuple[float, float]:
    return (float(np.mean([float(rows[s]["completion_rate"]) for s in SEEDS])),
            float(np.mean([float(rows[s]["failure_adjusted_error_m"]) for s in SEEDS])))


def decide() -> dict[str, Any]:
    """The stage-3 selection rule, applied to seed-mean validation scores."""
    val = validation()
    score = {arm: seed_mean(val[arm]) for arm in CANDIDATES}
    c_d, e_d = score[GSD]
    c_w, e_w = score[GSDW]
    wide = c_w > c_d + 1e-12 or (abs(c_w - c_d) <= 1e-12 and e_w < e_d)
    box_arm, long_arm = (GSDW, GSDW2M) if wide else (GSD, GSD2M)
    c_b, e_b = score[box_arm]
    c_l, e_l = score[long_arm]
    longer = (c_l - c_b >= 0.02 - 1e-12) or (abs(c_l - c_b) <= 1e-12 and e_l <= 0.9 * e_b)
    chosen = long_arm if longer else box_arm
    return {
        "score": score, "wide": wide, "longer": longer, "chosen_d": chosen,
        "chosen_n": GSNW if wide else GSN, "chosen_h10": GSDWH10 if wide else GSDH10,
        "box_arm": box_arm, "long_arm": long_arm,
    }


DEC = decide()
CHOSEN_D, CHOSEN_N, CHOSEN_H10 = DEC["chosen_d"], DEC["chosen_n"], DEC["chosen_h10"]
MAIN = ["pid_global_nominal", "pid_global_robust", PIDRN, CHOSEN_N, CHOSEN_D, DRN, DRD]
NEW = [GSDW, GSNW, GSDWH10, GSD2M, GSDW2M]
TABLE_ARMS = MAIN + [a for a in [GSD, GSDW, GSD2M, GSDW2M, GSN, GSNW, GSDH10, GSDWH10] if a not in MAIN]


# ---------------------------------------------------------------- sections

def run_record() -> str:
    names = ["g0_occupancy", "replay_plan2"] + [f"g4_{s}_{t}" for t in ("g1", "g2") for s in
                                                 ("heldout_ranged", "heldout_fixed", "delay", "dynamic", "noise")]
    names[7:7] = ["g4_frozen_eval", "g4_occupancy_wide"]
    rows = []
    for name in names:
        m = meta(P3 / name / "metadata.json")
        machine = m["machine"]
        rows.append([f"`{name}`", machine.get("slurm_job_id", "?"), machine.get("node", "?").split(".")[0],
                     f"`{m['git']['commit'][:7]}`" + (" dirty" if m["git"].get("dirty") else ""),
                     m.get("episodes", m.get("jobs", "–")), f"{m['wall_time_s'] / 60:.1f}"])
    g = meta(P3 / "g4_frozen_gains" / "frozen_gains.json")
    rows.insert(7, ["`g4_frozen_gains`", g["machine"].get("slurm_job_id"), g["machine"]["node"].split(".")[0],
                    f"`{g['git']['commit'][:7]}`", "–", f"{g['wall_time_s'] / 60:.1f}"])
    rc = meta(P3 / "replay_plan2" / "replay_check.json")
    m = meta(P3 / "g4_heldout_ranged_g1" / "metadata.json")
    versions = ", ".join(f"{k} {v}" for k, v in sorted(m.get("packages", {}).items()))
    return "\n\n".join([
        "All stage-3 work ran in SLURM job 3281395 on cpunode13 (commit `8e38f2f`).",
        table(["output", "SLURM job", "node", "commit", "episodes / jobs", "wall min"], rows),
        f"- **Machine:** {m['machine']['processor']}, {m['machine']['platform']}, Python {m['machine']['python']}.",
        f"- **Packages:** {versions}.",
        "- **Parity gate:** `check_parity.py` printed `PARITY OK` at the start of the job.",
        f"- **Replay check:** with the stage-3 evaluation code (per-checkpoint gain box), GS-D, GS-N and DR-D "
        f"(stage 2) were re-run on the ranged held-out set. {rc['bit_identical_rows']} of {rc['rows_checked']} "
        "rows are bit-identical to the stage-2 rows (inference-latency columns excluded). All other stage-2 rows "
        "used below are taken from `data/results/stage2/`.",
    ])


def box_section() -> str:
    from pathfollow.calibration import GainCalibration, PIDGains

    dt = 1.0 / config.CONFIG.policy_hz
    rows = []
    for label, path, rate in (("stage 2 (GS-D, GS-N)", "blind_ppo.json", None),
                              ("stage 3 wide (…-W)", "blind_ppo_wide.json", (147.5, 0.75, 1.75))):
        c = GainCalibration.load(config.CALIBRATION / path)
        lo, hi, g0 = c.lower.as_array(), c.upper.as_array(), c.nominal.as_array()
        r = np.asarray(rate) if rate else 0.5 * (hi - lo)
        for i, g in enumerate(("Kp", "Ki", "Kd")):
            rows.append([label if i == 0 else "", f"`{path}`" if i == 0 else "", g, f"{lo[i]:g}", f"{hi[i]:g}",
                         f"{g0[i]:g}", f"{g0[i] - lo[i]:g}", f"{hi[i] - g0[i]:g}", f"{r[i]:g}", f"{r[i] * dt:g}"])
    old = GainCalibration.load(config.CALIBRATION / "blind_ppo.json")
    wide = GainCalibration.load(config.CALIBRATION / "blind_ppo_wide.json")
    where = []
    for name, gains in (("PID-RN", (2.0, 0.5, 0.5)), ("PID-R", (8.0, 0.5, 4.4))):
        for label, c in (("stage 2", old), ("wide", wide)):
            lo, hi = c.lower.as_array(), c.upper.as_array()
            status = ", ".join(f"{g} {'inside' if lo[i] <= v <= hi[i] else 'outside'}"
                               for i, (g, v) in enumerate(zip(("Kp", "Ki", "Kd"), gains)))
            action = c.action_for(PIDGains(*gains)) if all(lo <= np.asarray(gains)) else None
            where.append([name, label, status, "–" if action is None else
                          ", ".join(f"{a:+.3f}" for a in action)])
    return "\n\n".join([
        "**Map** (`calibration.GainCalibration.map_action`): action a ∈ [−1, 1] per gain, piecewise linear "
        "around g0 — gain = g0 + a·(g0 − lower) for a < 0, g0 + a·(upper − g0) for a ≥ 0. The policy acts at "
        f"50 Hz (one decision = {dt:g} s). Applied gains move towards the target by at most the rate limit per "
        "decision and start at g0 in every episode. The derivative filter is fixed at "
        f"τd = {config.CONFIG.derivative_filter_tau_s:g} s for every scheduler (the same value as PID-RN).",
        table(["box", "file", "gain", "lower", "upper", "g0", "lower span", "upper span", "rate limit /s",
               "max change / decision"], rows),
        "**Baseline PIDs relative to the box** (action = the scheduler action that targets those gains)",
        table(["PID", "box", "per gain", "action (Kp, Ki, Kd)"], where),
        "With the stage-3 rate limit (kept at the stage-2 values), moving from g0 to PID-RN's gains takes "
        "(26 − 2) / 147.5 = 0.16 s for Kp and (4.4 − 0.5) / 1.75 = 2.2 s for Kd. Under the plan's \"half the range\" "
        "rule the Kd limit would have been 3.0 /s (+71 %); see §0.1.",
    ])


def occupancy_tables(directory: str, arms: list[str]) -> tuple[str, str]:
    rows = read_csv(P3 / directory / "occupancy.csv")
    pooled = defaultdict(list)
    for row in rows:
        pooled[(row["controller"], row["manifest"], row["condition"])].append(row)
    keys = ["kp_target_at_lower_edge", "kp_applied_on_lower_bound", "kd_target_at_lower_edge",
            "kd_applied_on_lower_bound", "ki_target_at_lower_edge", "kp_target_at_upper_edge"]
    main = []
    for (arm, manifest, condition), group in sorted(pooled.items(), key=lambda kv: (arms.index(kv[0][0]), kv[0][1:])):
        steps = np.asarray([float(g["policy_steps"]) for g in group])
        weighted = lambda k: float(np.sum(steps * np.asarray([float(g[k]) for g in group])) / steps.sum())  # noqa: E731
        main.append([SHORT[arm], manifest, condition] + [pct(weighted(k)) for k in keys]
                    + [f"{np.median([float(g['kp_applied_median']) for g in group]):.2f}",
                       f"{np.median([float(g['kd_applied_median']) for g in group]):.2f}"])
    per_seed = []
    for row in sorted(rows, key=lambda r: (arms.index(r["controller"]), r["manifest"], r["condition"],
                                           int(r["training_seed"]))):
        per_seed.append([SHORT[row["controller"]], row["manifest"], row["condition"], row["training_seed"]]
                        + [pct(row[k]) for k in keys]
                        + [f"{float(row[f'{g}_applied_{s}']):.2f}" for g in ("kp", "ki", "kd")
                           for s in ("mean", "median", "p5", "p95")])
    head = ["Kp target at lower edge %", "Kp applied on lower bound %", "Kd target at lower edge %",
            "Kd applied on lower bound %", "Ki target at lower edge %", "Kp target at upper edge %"]
    return (
        table(["arm", "manifest", "condition"] + head + ["median Kp (seed median)", "median Kd (seed median)"], main),
        table(["arm", "manifest", "condition", "seed"] + head
              + [f"{g} {s}" for g in ("Kp", "Ki", "Kd") for s in ("mean", "median", "p5", "p95")], per_seed),
    )


def history(root: Path, arm: str, seed: str) -> list[dict[str, float]]:
    path = root / arm / f"seed_{seed}" / "validation_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best(records: list[dict[str, float]]) -> dict[str, float]:
    chosen, key = None, None
    for record in records:
        k = (record["completion_rate"], -record["failure_adjusted_error_m"])
        if key is None or k > key:
            chosen, key = record, k
    return chosen


def curves_section() -> str:
    val = validation()
    rows = []
    for arm, root in ((GSD, M2), (GSN, M2), (DRD, M2), (DRN, M2), (GSDW, M3), (GSNW, M3), (GSDWH10, M3),
                      (GSD2M, M3), (GSDW2M, M3)):
        for s in SEEDS:
            h = history(root, arm, s)
            last = h[-1]
            sel = val[arm][s]
            rows.append([SHORT[arm], s, len(h), f"{int(last['timesteps']) // 1000}k", f"{int(sel['selected_timesteps']) // 1000}k",
                         f"{pct(sel['completion_rate'])} / {mm(sel['failure_adjusted_error_m'])}",
                         f"{pct(last['completion_rate'])} / {mm(last['failure_adjusted_error_m'])}"])
    return table(["arm", "seed", "checkpoints", "last step", "selected step", "selected val compl % / E_FA mm",
                  "last checkpoint compl % / E_FA mm"], rows)


def g2_section() -> str:
    rows, same = [], []
    for long_arm, short_arm, root_short in ((GSD2M, GSD, M2), (GSDW2M, GSDW, M3)):
        for s in SEEDS:
            h = history(M3, long_arm, s)
            at = min(h, key=lambda r: abs(r["timesteps"] - 300_000))
            upto = best([r for r in h if r["timesteps"] <= 300_000])
            short_sel = best(history(root_short, short_arm, s))
            full = best(h)
            match = (upto["completion_rate"], upto["failure_adjusted_error_m"]) == (
                short_sel["completion_rate"], short_sel["failure_adjusted_error_m"])
            same.append(match)
            rows.append([SHORT[long_arm], s, f"{int(at['timesteps']) // 1000}k",
                         f"{pct(at['completion_rate'])} / {mm(at['failure_adjusted_error_m'])}",
                         f"{pct(upto['completion_rate'])} / {mm(upto['failure_adjusted_error_m'])} @ {int(upto['timesteps']) // 1000}k",
                         f"{pct(full['completion_rate'])} / {mm(full['failure_adjusted_error_m'])} @ {int(full['timesteps']) // 1000}k",
                         "yes" if match else "no"])
    return "\n\n".join([
        "**Schedule.** PPO hyperparameters are those of the scheduler profile (`controller_profiles.py`): constant "
        "learning rate 3e-4 and constant entropy coefficient, so no schedule depends on the total step count. "
        "Only `--scheduler-timesteps 2000000` changes. The checkpoint interval stays at 25k steps (80 "
        "validations per run, the same 12 per 300k as before).",
        "**Training cost**",
        f"**GS-D 2M**\n\n{training_cost(P3 / 'training_ranged_h1_2m.log')}",
        f"**GS-D-W 2M**\n\n{training_cost(P3 / 'training_ranged_h1_wide_2m.log')}",
        "**Validation at the 300k point** (`validation_ranged.json`, completion % / E_FA mm)",
        table(["arm", "seed", "checkpoint nearest 300k", "score there", "best up to 300k", "best over 2M",
               "best-up-to-300k equals the 300k run's selection"], rows),
        f"- **Same selection as the 300k runs:** in {sum(same)} of {len(same)} runs, the best checkpoint up to "
        "300k has the same validation score as the checkpoint kept by the corresponding 300k run (same seed and "
        "settings, deterministic training).",
    ])


def decision_section() -> str:
    d = DEC
    rows = []
    val = validation()
    for arm in CANDIDATES:
        c, e = d["score"][arm]
        rows.append([SHORT[arm], *[f"{pct(val[arm][s]['completion_rate'])} / {mm(val[arm][s]['failure_adjusted_error_m'])}"
                                   for s in SEEDS], f"**{pct(c)} / {mm(e)}**"])
    c_d, e_d = d["score"][GSD]
    c_w, e_w = d["score"][GSDW]
    c_b, e_b = d["score"][d["box_arm"]]
    c_l, e_l = d["score"][d["long_arm"]]
    return "\n\n".join([
        table(["candidate", *[f"s{s}" for s in SEEDS], "seed-mean"], rows),
        "Validation on `validation_ranged.json` (24 checkpoint scenarios), completion % / E_FA mm of the kept "
        "checkpoint. Rule fixed in the stage-3 plan before any result was seen.",
        f"- **Rule 1 (box):** GS-D-W {pct(c_w)} % vs GS-D {pct(c_d)} % seed-mean completion → "
        + ("**wide box chosen**; GS-N-W and GS-D-W H10 replace GS-N and GS-D H10." if d["wide"] else
           "**box stays [5, 300]**; the wide-box arms are reported as a check."),
        f"- **Rule 2 (budget):** {SHORT[d['long_arm']]} {pct(c_l)} % / {mm(e_l)} mm vs {SHORT[d['box_arm']]} "
        f"{pct(c_b)} % / {mm(e_b)} mm (difference {100 * (c_l - c_b):+.1f} pp; threshold +2 pp, or equal "
        "completion and E_FA ≥ 10 % lower) → "
        + (f"**{SHORT[d['long_arm']]} is the main scheduler.** The nominal and H10 scheduler arms stay at 300k "
           "(not retrained at 2M; a stated limitation)." if d["longer"] else
           f"**{SHORT[d['box_arm']]} (300k) stays**; more budget did not help on validation."),
        f"- **Main arms from here on:** {', '.join(SHORT[a] for a in MAIN)}. H10 ablation: {SHORT[CHOSEN_H10]} "
        f"(300k).",
    ])


def heldout_section() -> str:
    rows = load("heldout_ranged")
    ep = load("heldout_ranged", "episodes.csv")
    by_seed = load("heldout_ranged", "summary_by_seed.csv")
    idx, idx_seed = index(rows), index(by_seed, with_seed=True)
    fixed = index(load("heldout_fixed"))
    cells = r2.HELDOUT_CELLS
    versions = [(GSDW, GSD, True), (GSD2M, GSD, True), (GSDW2M, GSD, True)]
    vs = [(CHOSEN_D, PIDRN, False), (CHOSEN_D, DRD, True), (CHOSEN_D, "pid_global_robust", False)]
    extra = []
    if DEC["wide"]:
        extra = [(CHOSEN_N, GSN, True), (GSDW, GSDWH10, True)]
    v_err, v_comp = r2.paired_block(versions, ep, cells)
    c_err, c_comp = r2.paired_block(vs, ep, cells)
    blocks = ["### 5.2 Scheduler versions against stage-2 GS-D (matched seeds)",
              "**ΔE_FA (mm)**", v_err, "**ΔCompletion (pp)**", v_comp,
              f"### 5.3 {SHORT[CHOSEN_D]} against the baselines",
              "**ΔE_FA (mm)**", c_err, "**ΔCompletion (pp)**", c_comp]
    if extra:
        e_err, e_comp = r2.paired_block(extra, ep, cells)
        blocks += ["### 5.4 Wide-box nominal scheduler and history (H1 − H10 on the wide box, both 300k)",
                   "**ΔE_FA (mm)**", e_err, "**ΔCompletion (pp)**", e_comp]
    return "\n\n".join([
        "### 5.1 Ranged held-out 480 (completion % / E_FA mm, seed-pooled)",
        r1.heldout_table(idx, TABLE_ARMS),
        details("All metrics", r2.heldout_metric_table(idx, TABLE_ARMS)),
        details("Per seed (scheduler arms)", r1.heldout_seed_table(idx_seed, [a for a in TABLE_ARMS if "scheduler" in a])),
        *blocks,
        "### 5.5 Fixed-severity held-out (0.15 s / 0.3 mm), for comparison",
        r1.heldout_table(fixed, TABLE_ARMS),
    ])


SWEEP_ARMS = MAIN + [a for a in [GSD, GSDW, GSD2M, GSDW2M, GSDWH10, GSNW, GSN] if a not in MAIN]


def sweeps_section() -> str:
    rows = load("delay")
    idx, levels = index(rows), levels_of(rows)
    drows = load("dynamic")
    didx, dlevels = index(drows), levels_of(drows)
    dep = load("dynamic", "episodes.csv")
    dyn = []
    for variant in ("delay", "combined"):
        body = table(["D_max"] + [SHORT[a] for a in SWEEP_ARMS],
                     ([f"{lv:g} s"] + [cell(didx.get((a, lv, "stationary", variant))) for a in SWEEP_ARMS]
                      for lv in dlevels))
        dyn.append(f"**{variant}** (completion % / E_FA mm, seed-pooled)\n\n{body}")
    pairs = [(GSDW, GSD, True), (GSD2M, GSD, True), (GSDW2M, GSD, True),
             (CHOSEN_D, PIDRN, False), (CHOSEN_D, DRD, True), (CHOSEN_D, "pid_global_robust", False)]
    if DEC["wide"]:
        pairs += [(GSDW, GSDWH10, True)]
    blocks = []
    for lv in dlevels:
        e, c = r2.paired_block(pairs, dep, [("stationary", "delay"), ("stationary", "combined")], level=lv)
        blocks.append(f"**D_max = {lv:g} s, ΔE_FA (mm)**\n\n{e}\n\n**D_max = {lv:g} s, ΔCompletion (pp)**\n\n{c}")
    return "\n\n".join([
        "### 6.1 Constant-delay sweep 0–0.8 s (F3 construction)",
        sweep_tables(idx, SWEEP_ARMS, levels, "delay", ("delay",)),
        "**Breaking points** (E1 rule)",
        breaking_table(idx, SWEEP_ARMS, levels, ("delay",), "delay"),
        details("All metrics", full_metric_table(idx, SWEEP_ARMS, levels, "delay", ("delay",))),
        "### 6.2 Dynamic delay (F4 construction)",
        *dyn,
        details("All metrics", full_metric_table(didx, SWEEP_ARMS, dlevels, "delay", ("delay", "combined"))),
        "**Paired differences**", *blocks,
    ])


def noise_section() -> str:
    rows = load("noise")
    idx, levels = index(rows), levels_of(rows)
    return "\n\n".join([
        "F5 construction (coarse + fine; noise only and noise + 0.15 s delay; both modes). PID-N, PID-R, GS-N H10 and "
        "DR-N H10 rows are the E1 rows; stage-2 arms are the stage-2 rows.",
        sweep_tables(idx, SWEEP_ARMS, levels, "noise", ("noise", "combined")),
        "**Breaking points** (E1 rule)",
        breaking_table(idx, SWEEP_ARMS, levels, ("noise", "combined"), "noise"),
        details("All metrics incl. mechanism columns (steering TV/T, wheel saturation)",
                full_metric_table(idx, SWEEP_ARMS, levels, "noise", ("noise", "combined"))),
    ])


def frozen_section() -> str:
    gains = meta(P3 / "g4_frozen_gains" / "frozen_gains.json")["gains"]
    g_rows = []
    for arm in (GSDW, GSNW):
        for s in SEEDS:
            v = gains[arm][s]
            g_rows.append([SHORT[arm], s] + [
                f"{v[k]['mean']['kp']:.2f} / {v[k]['mean']['ki']:.3f} / {v[k]['mean']['kd']:.2f}" for k in ("clean", "delay")])
    ep = read_csv(P3 / "g4_frozen_eval" / "episodes.csv")
    c_rows = []
    for arm in (GSDW, GSNW):
        per = {v: r2.seed_completion(ep, name, r2.DELAY_CELLS) for v, name in
               (("dyn", arm), ("clean", f"{arm}_frozen_clean"), ("delay", f"{arm}_frozen_delay"))}
        for s in SEEDS + ("mean",):
            vals = {v: (np.mean(list(p.values())) if s == "mean" else p[s]) for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            c_rows.append([SHORT[arm], s, pct(vals["dyn"]), pct(vals["clean"]), pct(vals["delay"]),
                           f"{gap * 100:+.1f}", "**yes**" if s != "mean" and gap > 0.05 else ""])
    occ2, occ2_seed = occupancy_tables("g4_occupancy_wide", [GSDW, GSNW])
    return "\n\n".join([
        "### 8.1 Frozen gains (F6 construction: gains from `validation.json`, mean; evaluated on the ranged held-out "
        "set, each with its own box and rate limit)",
        table(["arm", "seed", "F-clean Kp / Ki / Kd", "F-delay Kp / Ki / Kd"], g_rows),
        "**Completion in the four delay-bearing cells (%)**",
        table(["arm", "seed", "dynamic", "F-clean", "F-delay", "dynamic − best frozen (pp)", "flag (> 5 pp)"], c_rows),
        "### 8.2 Gain occupancy of the wide-box schedulers",
        occ2, details("Per seed", occ2_seed),
        "Figure: `figures/g4_gain_occupancy_wide.{pdf,png}`.",
    ])


# ---------------------------------------------------------------- G5 (addendum: lookups, no training)

EVENT_METRICS = (("post_event_peak_distance_m", "post-event peak path distance", 1000, "mm"),
                 ("post_event_excess_iae_m_s", "post-event excess IAE", 1000, "mm·s"))
EVENT_CELLS = [("transient", c) for c in ("delay", "noise", "combined")]


def paired_metric(rows, a, b, mode, condition, match_seed, key):
    """Seed -> arrays of (a - b) for one metric, scenarios where both rows define it."""
    ref = {}
    for row in rows:
        if row["controller"] == b and row["evaluation_mode"] == mode and row["condition"] == condition:
            ref[(row["training_seed"] if match_seed else "", row["scenario_id"])] = row
    out = defaultdict(list)
    for row in rows:
        if row["controller"] != a or row["evaluation_mode"] != mode or row["condition"] != condition:
            continue
        base = ref[(row["training_seed"] if match_seed else "", row["scenario_id"])]
        if row[key] in ("", None) or base[key] in ("", None):
            continue
        out[row["training_seed"] or "fixed"].append(float(row[key]) - float(base[key]))
    return {k: np.asarray(v) for k, v in out.items()}


def g5_section() -> str:
    ep = load("heldout_ranged", "episodes.csv")
    rows = [r for r in ep if r["controller"] in MAIN and r["evaluation_mode"] == "transient"]
    # 1. event metrics, seed-pooled means over every episode that defines them (as the sealed summary)
    table_rows = []
    for arm in MAIN:
        line = [SHORT[arm]]
        for key, _, scale, _unit in EVENT_METRICS:
            for _, condition in EVENT_CELLS:
                vals = [float(r[key]) for r in rows if r["controller"] == arm and r["condition"] == condition
                        and r[key] not in ("", None)]
                line.append(f"{np.mean(vals) * scale:.1f}" if vals else "–")
        table_rows.append(line)
    header = ["arm"] + [f"{label.split(' ')[1]} {c} ({unit})" for _, label, _, unit in EVENT_METRICS
                        for _, c in EVENT_CELLS]
    pair_rows = []
    for a, b, match in ((DRD, PIDRN, False), (CHOSEN_D, PIDRN, False)):
        for key, label, scale, unit in EVENT_METRICS:
            for mode, condition in EVENT_CELLS:
                values = paired_metric(ep, a, b, mode, condition, match, key)
                mean, low, high = r1.estimate(values)
                pair_rows.append([f"{SHORT[a]} − {SHORT[b]}", label, condition]
                                 + [f"{np.mean(values[s]) * scale:+.2f}" if s in values else "–" for s in SEEDS]
                                 + [f"**{mean * scale:+.2f}** {unit}", f"[{low * scale:+.2f}, {high * scale:+.2f}]"])
    # 2. onset and severity
    book = r2.bookkeeping("held_out_ranged")
    manifest = json.loads((r2.MAN / "held_out_ranged.json").read_text(encoding="utf-8"))["scenarios"]
    trans = [s for s in manifest if s["evaluation_mode"] == "transient"]
    starts = sorted({e["start_s"] for s in trans for e in s["events"]})
    clean = all(s["initial_delay_s"] == 0 and s["initial_noise_std_m"] == 0 for s in trans)
    d = np.array([book[s["scenario_id"]]["delay_s"] for s in trans if s["condition"] in ("delay", "combined")])
    n = np.array([book[s["scenario_id"]]["noise_m"] for s in trans if s["condition"] in ("noise", "combined")]) * 1000
    # 3. inference time per control decision (cluster, one torch thread per worker process)
    inf_rows = []
    for arm in MAIN:
        r = [x for x in ep if x["controller"] == arm]
        means = np.asarray([float(x["mean_inference_ms"]) for x in r])
        p99 = np.asarray([float(x["p99_inference_ms"]) for x in r])
        decisions = np.asarray([float(x["policy_decisions"]) for x in r])
        inf_rows.append([SHORT[arm], len(r), f"{np.sum(means * decisions) / np.sum(decisions) * 1000:.1f}",
                         f"{np.median(p99) * 1000:.1f}", f"{np.percentile(p99, 95) * 1000:.1f}"])
    meta_g4 = meta(P3 / "g4_heldout_ranged_g2" / "metadata.json")
    from pathfollow.core.dynamics import DT

    obs = observation_table()
    return "\n\n".join([
        "### 9.1 Transient event metrics, ranged held-out (transient cells)",
        "Definitions (`env.py::_recovery_metrics`, unchanged since the sealed run):\n"
        "- **Event onset:** the earliest event start in the scenario.\n"
        "- **Baseline:** the mean path distance over the 1 s before onset.\n"
        "- **Post-event peak path distance:** the maximum path distance in [onset, onset + 2 s].\n"
        "- **Post-event excess IAE:** the integral of max(distance − baseline, 0) over the same 2 s window.\n"
        "- **Aggregation:** seed-pooled means over every episode that defines them, as in the summary tables. "
        "Failed episodes are included if they reach the onset.",
        table(header, table_rows),
        "**Paired differences** (hierarchical bootstrap, 10,000 replicates, seed-mean; learned seeds against the "
        "single PID instance)",
        table(["pair", "metric", "transient cell"] + [f"s{s}" for s in SEEDS] + ["seed-mean", "95 % CI"], pair_rows),
        "### 9.2 Transient onset and severity in the ranged held-out set",
        f"- **Onset:** every transient record has its events at {', '.join(f'{s:g} s' for s in starts)} "
        f"(`config.transient_start_s`). Before onset the scenario is clean: "
        f"{'yes, no initial delay or noise in any transient record' if clean else 'NO'}.\n"
        "- **Event structure:** transient delay and noise records carry one step event. Combined records carry "
        "two, a delay step and a noise step, both at the onset.\n"
        f"- **Severity:** drawn once per record with the generator keyed on (seed 20261116, record index). "
        f"Delay is uniform in [0, 0.3] s (transient draws: min {d.min():.3f}, median {np.median(d):.3f}, "
        f"max {d.max():.3f} s). Noise is log-uniform in [0.1, 10] mm (transient draws: min {n.min():.2f}, median "
        f"{np.median(n):.2f}, max {n.max():.2f} mm). The step value is the drawn severity, and it holds to the "
        "end of the episode.",
        "### 9.3 Run-time cost per control decision",
        table(["arm", "episodes", "mean (µs)", "median of per-episode p99 (µs)",
               "95th pct of per-episode p99 (µs)"], inf_rows),
        f"- **What is measured:** wall time of one `predictor(observation)` call, i.e. the policy forward pass, "
        f"or for the fixed PIDs the constant action. It was measured in `rollout.run_episode` with "
        f"`time.perf_counter_ns` during the ranged held-out run on {meta_g4['machine']['processor']}. Each of the "
        f"{meta_g4['workers']} worker processes ran with one torch thread, so each measurement is on one core.\n"
        "- **Not included:** the PID law itself and the gain-scheduler map run inside the environment step "
        f"(physics at {1 / DT:g} Hz, {config.CONFIG.frame_skip} substeps per decision).\n"
        "- **Statistics recorded:** per episode, the mean and the 99th percentile. The plan's 95th percentile per "
        "decision was not recorded. The table reports the decision-weighted mean and the spread of the "
        "per-episode 99th percentiles.\n"
        "- **Training wall time:** stage-2 runs in `stage2.md` §2; stage-3 runs in §2–3 of this document.",
        "### 9.4 Observation contents (one frame, H = 1)",
        obs,
    ])


def observation_table() -> str:
    from pathfollow.controller_profiles import DIRECT_FRAME_FIELDS, SCHEDULER_FRAME_FIELDS

    c = config
    vmax = max(c.CONFIG.target_speeds)
    common = {
        "e_ct_rate": ("rate of the measured cross-track error over the last decision (m/s)",
                      f"÷ {c.E_CT_RATE_SCALE_MPS:g} m/s"),
        "e_theta": ("heading error to the path tangent (rad)", "÷ π"),
        "speed": ("forward speed of the chassis (m/s)", f"÷ {c.SPEED_SCALE_MPS:g} m/s"),
        "yaw_rate": ("yaw rate (rad/s)", f"÷ {c.YAW_RATE_SCALE_RADPS:g} rad/s"),
        "omega_applied": ("steering command after wheel allocation (normalised wheel units)", "already in [−1, 1]"),
        "v_applied": ("speed command after wheel allocation (normalised wheel units)", "already in [−1, 1]"),
        "v_target": ("target speed of the episode (m/s)", f"÷ {vmax:g} m/s"),
        "wheel_utilization": ("larger absolute value of the two allocated wheel commands", "already in [0, 1]"),
        "allocation_limited": ("1 if allocation had to change the requested (v, ω), else 0", "flag"),
    }
    gs = dict(common)
    gs.update({
        "e_ct": ("measured (noisy) cross-track error (m)", f"÷ {c.E_CT_SCALE_M:g} m"),
        "e_ct_integral": ("integral state of the steering PID (m·s)", f"÷ {c.INTEGRAL_SCALE_MS:g} m·s"),
        "omega_requested": ("steering PID output before allocation (normalised)", "PID output limits [−1, 1]"),
        "previous_action_kp": ("previous scheduler action for Kp (action coordinates)", "already in [−1, 1]"),
        "previous_action_ki": ("previous scheduler action for Ki", "already in [−1, 1]"),
        "previous_action_kd": ("previous scheduler action for Kd", "already in [−1, 1]"),
        "pid_saturated": ("1 if the unclamped PID output exceeds 1, else 0", "flag"),
        "steer_integrator_hold": ("1 if the PID integrator was held (anti-windup) at the last substep, else 0",
                                  "flag"),
    })
    dr = dict(common)
    dr.update({
        "e_ct": ("measured (noisy) cross-track error (m)", "÷ 0.2 m"),
        "e_ct_integral": ("environment-side error integral, itself clipped to ±1 m·s (m·s)",
                          f"÷ {c.INTEGRAL_SCALE_MS:g} m·s"),
        "previous_steer": ("previous policy action (steering command)", "already in [−1, 1]"),
        "steer_saturated": ("1 if the absolute steering action reaches 1, else 0", "flag"),
    })
    parts = []
    for title, fields, meaning in (("Gain scheduler (GS), 17 fields", SCHEDULER_FRAME_FIELDS, gs),
                                   ("Direct policy (DR), 13 fields", DIRECT_FRAME_FIELDS, dr)):
        rows = [[i + 1, f"`{f}`", *meaning[f]] for i, f in enumerate(fields)]
        parts.append(f"**{title}** (order as in the observation vector)\n\n"
                     + table(["#", "field", "content (unit)", "scaling"], rows))
    return "\n\n".join(parts) + (
        "\n\n- **Clipping:** every field is clipped to [−1, 1] after scaling (`env.py::_frame`).\n"
        "- **Timing:** fields are taken at the last physics substep of a decision. The frame is built every 20 ms "
        "(50 Hz).\n"
        "- **History:** with H = 1 the observation is this single frame. The previous action (GS: three gain "
        "actions; DR: previous steer) is the only memory. With H = 10, the last ten frames are concatenated, "
        "oldest first.\n"
        "- **Differences between the two policies:** the GS and DR frames use different e_ct scales (1 m vs 0.2 m) "
        "and different integral sources (the PID's internal integral vs an environment-side integral). These "
        "come from their separate controller profiles.")


def frozen_tables(gains_dir: str, eval_dir: str, arms: list[str]) -> tuple[str, str]:
    gains = meta(P3 / gains_dir / "frozen_gains.json")["gains"]
    g_rows = []
    for arm in arms:
        for s in SEEDS:
            v = gains[arm][s]
            g_rows.append([SHORT[arm], s] + [
                f"{v[k]['mean']['kp']:.2f} / {v[k]['mean']['ki']:.3f} / {v[k]['mean']['kd']:.2f}" for k in ("clean", "delay")])
    ep = read_csv(P3 / eval_dir / "episodes.csv")
    c_rows = []
    for arm in arms:
        per = {v: r2.seed_completion(ep, name, r2.DELAY_CELLS) for v, name in
               (("dyn", arm), ("clean", f"{arm}_frozen_clean"), ("delay", f"{arm}_frozen_delay"))}
        for s in SEEDS + ("mean",):
            vals = {v: (np.mean(list(p.values())) if s == "mean" else p[s]) for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            c_rows.append([SHORT[arm], s, pct(vals["dyn"]), pct(vals["clean"]), pct(vals["delay"]),
                           f"{gap * 100:+.1f}", "**yes**" if s != "mean" and gap > 0.05 else ""])
    return (table(["arm", "seed", "F-clean Kp / Ki / Kd", "F-delay Kp / Ki / Kd"], g_rows),
            table(["arm", "seed", "dynamic", "F-clean", "F-delay", "dynamic − best frozen (pp)", "flag (> 5 pp)"],
                  c_rows))


def g12_section() -> str:
    rows = []
    for name, file in (("g5_occupancy_2m", "metadata.json"), ("g5_frozen_gains_2m", "frozen_gains.json"),
                       ("g5_frozen_eval_2m", "metadata.json")):
        m = meta(P3 / name / file)
        rows.append([f"`{name}`", m["machine"].get("slurm_job_id", "?"), m["machine"]["node"].split(".")[0],
                     f"`{m['git']['commit'][:7]}`" + (" dirty" if m["git"].get("dirty") else ""),
                     m.get("episodes", m.get("jobs", "–")), f"{m['wall_time_s'] / 60:.1f}"])
    rc = meta(P3 / "g5_frozen_eval_2m" / "replay_check.json")
    occ, occ_seed = occupancy_tables("g5_occupancy_2m", [GSDW2M])
    gains, completion = frozen_tables("g5_frozen_gains_2m", "g5_frozen_eval_2m", [GSDW2M])
    return "\n\n".join([
        "Follow-up job `cluster/stage3b_selected_scheduler_gains.sh`: replays the five GS-D-W 2M checkpoints (the main scheduler, thesis name "
        "\"GS-D\") with the same tools and settings as §1.2 and §8. No training.",
        table(["output", "SLURM job", "node", "commit", "episodes / jobs", "wall min"], rows),
        f"- **Machine:** AMD EPYC 9334, 16 cores. `check_parity.py` printed `PARITY OK`.\n"
        f"- **Replay check:** the dynamic GS-D-W 2M rows of this job reproduce {rc['bit_identical_rows']} of "
        f"{rc['rows_checked']} rows of `g4_heldout_ranged_g2` bit for bit.",
        "### 12.1 Gain occupancy, GS-D-W 2M (same definitions as §1.2: edges in action coordinates, the Kp lower edge "
        "is Kp ≤ 1.5 on the wide range; percentages step-weighted over seeds; medians = median of per-seed medians)",
        occ, details("Per seed (incl. mean / median / p5 / p95 of every gain)", occ_seed),
        "### 12.2 Frozen gains, GS-D-W 2M (same construction as §8.1: mean applied gains on `validation.json`, "
        "F-clean from nominal scenarios, F-delay from delay and combined scenarios, evaluated as constant gains on the "
        "ranged held-out set with the checkpoint's own gain range and rate limit)",
        gains,
        "**Completion in the four delay-bearing cells (%)**",
        completion,
    ])


def observations() -> dict[str, str]:
    path = NOTES / "stage3.md"
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
    ("g0_gain_occupancy", "G0: applied Kp of stage-2 GS-D on the ranged held-out set, per condition (log x), box bounds and PID-RN / PID-R Kp marked."),
    ("g0_learning_curves", "G0: validation completion and E_FA vs training steps, GS-D, GS-N, DR-D, DR-N (stage 2), seeds + mean, kept checkpoints marked."),
    ("g2_learning_curves_2m", "G2: the same for GS-D 2M and GS-D-W 2M; 300k marked."),
    ("g4_heldout_main", "G4: the seven main arms on the ranged held-out set; stage-2 GS-D in grey."),
    ("g4_delay_sweep", "G4: constant delay 0–0.8 s, main arms; stage-2 GS-D as a thin grey line."),
    ("g4_dynamic_delay", "G4: dynamic delay, main arms; stage-2 GS-D in grey."),
    ("g4_noise_sweep_stationary", "G4: noise sweep, stationary, main arms; stage-2 GS-D in grey."),
    ("g4_noise_sweep_transient", "G4: noise sweep, transient, main arms; stage-2 GS-D in grey."),
    ("g4_mechanism_stationary_noise", "G4: steering TV/T and wheel saturation vs noise, main arms; stage-2 GS-D in grey."),
    ("g4_gain_occupancy_wide", "G4: applied Kp of GS-D-W vs stage-2 GS-D on the ranged held-out set (log x), both boxes marked."),
]


def build() -> str:
    obs = observations()
    o = lambda key: obs.get(key, "_(observations not written yet)_")  # noqa: E731
    occ, occ_seed = occupancy_tables("g0_occupancy", [GSD, GSN])
    names = ("| short name | arm | box | budget | history |\n|---|---|---|---|---|\n"
             "| GS-D / GS-N | stage-2 schedulers, ranged / nominal sampler | [5, 300] | 300k | H = 1 |\n"
             "| GS-D-W / GS-N-W | wide-box schedulers | wide | 300k | H = 1 |\n"
             "| GS-D-W H10 | wide-box history ablation | wide | 300k | H = 10 |\n"
             "| GS-D 2M / GS-D-W 2M | budget-parity schedulers | [5, 300] / wide | 2M | H = 1 |\n"
             "| PID-N, PID-R, PID-RN, DR-N, DR-D, GS-D H10 | as in `stage2.md` | | | |")
    out = [
        "# Experiment Results 3",
        "Plan: the stage-3 plan (scheduler gain box and training budget). Generated by `report_stage3.py` from "
        "`data/results/stage3/` and the stage-2 results. Rates are seed-pooled unless marked; E_FA in mm.",
        "**Arm names**", names,
        f"**Outcome of the fixed decision rule (§4):** main scheduler = **{SHORT[CHOSEN_D]}**; nominal scheduler = "
        f"**{SHORT[CHOSEN_N]}**; H10 ablation = **{SHORT[CHOSEN_H10]}**.",
        "## 0. Run record", run_record(),
        "### 0.1 Deviations from the stage-3 plan", o("0 Deviations"),
        "## 1. G0 Diagnostics",
        "### 1.1 Box definition", box_section(),
        "### 1.2 Gain occupancy, stage-2 schedulers",
        "Edges are measured in action coordinates: a target counts as at the lower edge when its action is ≤ −0.98, "
        "i.e. within 2 % of the lower segment of the map (Kp ≤ 5.42, Kd ≤ 2.54 on the stage-2 box). \"Applied on bound\" "
        "is the fraction of policy steps whose applied gain equals the bound. Percentages are step-weighted over "
        "seeds; medians are the median of the five per-seed medians.",
        occ, details("Per seed (incl. mean / median / p5 / p95 of every gain)", occ_seed),
        "Figure: `figures/g0_gain_occupancy.{pdf,png}`.",
        "### 1.3 Learning curves (validation history of every run)", curves_section(),
        "Figures: `figures/g0_learning_curves.{pdf,png}`, `figures/g2_learning_curves_2m.{pdf,png}`.",
        o("G0"),
        "## 2. G1 Wide box",
        "Changed: the lower bounds only (`data/calibration/blind_ppo_wide.json`, Kp 5 → 1, Kd 2.5 → 0, Ki already "
        "0). Upper bounds, g0 = (26, 0.5, 4.4), the upper segment of the map, reward, sampler, hyperparameters, "
        "seeds, 300k budget and checkpoint rule are unchanged. The rate limit is passed explicitly "
        "(`train.py --gain-rate-limit 147.5,0.75,1.75`) so it stays at the stage-2 values (§1.1).",
        f"**GS-D-W**\n\n{training_cost(P3 / 'training_ranged_h1_wide.log')}",
        f"**GS-N-W**\n\n{training_cost(P3 / 'training_nominal_h1_wide.log')}",
        f"**GS-D-W H10**\n\n{training_cost(P3 / 'training_ranged_h10_wide.log')}",
        "All 25 stage-3 runs trained at once on 64 cores beside the G0 replays, so wall times are not comparable "
        "with stage 2.",
        "## 3. G2 Budget parity (2M steps)", g2_section(), o("G2"),
        "## 4. G3 Decision (validation only)", decision_section(),
        "## 5. G4 Ranged and fixed held-out", heldout_section(), o("G4 held-out"),
        "## 6. G4 Delay sweep and dynamic delay", sweeps_section(), o("G4 delay"),
        "## 7. G4 Noise sweep and mechanism", noise_section(), o("G4 noise"),
        "## 8. G4 Frozen gains and gain occupancy of the new arms", frozen_section(), o("G4 frozen"),
        "## 9. G5 Gaps for the thesis (addendum, lookups only)", g5_section(), o("G5"),
        "## 10. Observations", "Facts per section are written under each section above. No interpretation.",
        "## 11. Files",
        "- **Results:** `data/results/stage3/<step>/`.\n- **Models:** `data/checkpoints/`.\n"
        "- **Calibration:** `data/calibration/blind_ppo_wide.json`.\n- **Code:** `gain_occupancy.py`, "
        "`check_replay.py`, `report_stage3.py`, `figures_stage3.py`, plus the per-checkpoint box in "
        "`evaluate.py` / `frozen_gains.py`.\n- **Job:** `cluster/stage3_scheduler_selection.sh`.",
        "**Figures** (`data/results/stage3/figures/`, PDF + 300 dpi PNG; same colour/marker scheme as stage 2; "
        "the chosen scheduler versions take the GS-D / GS-N colours, the stage-2 GS-D is a thin grey line).",
        table(["file", "content"], ([f"`{n}`", c] for n, c in FIGURES)),
        "## 12. GS-D (2M) gain occupancy and frozen gains (follow-up)", g12_section(), o("G12"),
    ]
    return "\n\n".join(part for part in out if part) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPORTS / "stage3.md"))
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
