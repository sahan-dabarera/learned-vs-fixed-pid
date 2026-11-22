"""Build POSTHOC_RESULTS_5.md (T02 replays) from artifacts/results/posthoc5
(cluster) or artifacts/results/posthoc5_local (provisional laptop run).

    T07  gain occupancy and frozen gains of GS-D / GS-N over ten seeds
         (seeds 11-71 from posthoc3, seeds 83-149 from this job)
    W18  mean / RMS / max |e_ct| per episode, main arms, ten seeds
    W17  per-step traces on three test paths; data note and representative seeds

    python posthoc_report5.py                      # uses posthoc5 when present, else posthoc5_local
    python posthoc_report5.py --results artifacts/results/posthoc5_local
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
import posthoc_report2 as r2
from posthoc_report import CONDITIONS, MODES, details, mm, pct, table
from metrics import read_csv

ROOT = Path(__file__).resolve().parent
P3 = ROOT / "artifacts" / "results" / "posthoc3"
P4 = ROOT / "artifacts" / "results" / "posthoc4"
THESIS_W17 = Path("D:/Msc/Thesis_Final/Experiments/w17")

OLD = ("11", "23", "37", "53", "71")
NEW = ("83", "97", "109", "127", "149")
ALL = OLD + NEW
GSD, GSN = "ppo_scheduler_ranged_h1_wide_2m", "ppo_scheduler_nominal_h1_wide"
DRD, DRN = "ppo_direct_ranged_h1", "ppo_direct_nominal_h1"
PIDN, PIDR, PIDRN = "pid_global_nominal", "pid_global_robust", "pid_global_robust_noise"
ARMS = [PIDN, PIDR, PIDRN, GSN, GSD, DRN, DRD]
LEARNED = [GSN, GSD, DRN, DRD]
NAME = {PIDN: "PID-N", PIDR: "PID-R", PIDRN: "PID-RN", GSN: "GS-N", GSD: "GS-D", DRN: "DR-N", DRD: "DR-D"}
PATHS = ("10019", "10001", "10014")
PATH_NOTE = {"10019": "gentlest (max |κ| 0.45 /m)", "10001": "median (max |κ| 1.18 /m)",
             "10014": "hardest (RMS κ 0.79 /m)"}
OCC_OLD = {GSD: "g5_occupancy_2m", GSN: "g4_occupancy_wide"}
FROZEN_OLD = {GSD: ("g5_frozen_gains_2m", "g5_frozen_eval_2m"), GSN: ("g4_frozen_gains", "g4_frozen_eval")}


def meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- run record

def run_record(res: Path) -> str:
    rows = []
    for name, file in (("occupancy_gsd_new", "metadata.json"), ("occupancy_gsn_new", "metadata.json"),
                       ("frozen_gains_new", "frozen_gains.json"), ("frozen_eval_new", "metadata.json"),
                       ("traces_heldout", "metadata.json"), ("traces_dynamic", "metadata.json")):
        m = meta(res / name / file)
        machine = m.get("machine", {})
        rows.append([f"`{name}`", machine.get("slurm_job_id") or "–", str(machine.get("node", "?")).split(".")[0],
                     f"`{m['git']['commit'][:7]}`" + (" dirty" if m["git"].get("dirty") else ""),
                     m.get("episodes", m.get("jobs", "–")), f"{m['wall_time_s'] / 60:.1f}"])
    checks = []
    for name in ("traces_heldout", "traces_dynamic"):
        for c in meta(res / name / "metadata.json").get("replay_checks", []):
            checks.append([name, c["rows_checked"], c["identical_rows"], c["unmatched_rows"]])
    rc = res / "frozen_eval_new" / "replay_check.json"
    if rc.is_file():
        c = meta(rc)
        checks.append(["frozen_eval_new (dynamic rows)", c["rows_checked"], c["bit_identical_rows"], 0])
    return "\n\n".join([
        table(["output", "SLURM job", "node", "commit", "episodes / jobs", "wall min"], rows),
        "**Replay check** (E_FA, finished and duration of each episode against the reported evaluation rows)",
        table(["output", "rows checked", "identical", "without a reference"], checks),
    ])


# ---------------------------------------------------------------- T07 occupancy

OCC_KEYS = ["kp_target_at_lower_edge", "kp_applied_on_lower_bound", "kd_target_at_lower_edge",
            "kd_applied_on_lower_bound", "ki_target_at_lower_edge", "kp_target_at_upper_edge"]
OCC_HEAD = ["Kp target at lower edge %", "Kp applied on lower bound %", "Kd target at lower edge %",
            "Kd applied on lower bound %", "Ki target at lower edge %", "Kp target at upper edge %"]


def occupancy_rows(res: Path, arm: str) -> list[dict[str, str]]:
    old = [r for r in read_csv(P3 / OCC_OLD[arm] / "occupancy.csv") if r["controller"] == arm]
    tag = "gsd" if arm == GSD else "gsn"
    new = [r for r in read_csv(res / f"occupancy_{tag}_new" / "occupancy.csv") if r["controller"] == arm]
    seeds = {r["training_seed"] for r in old + new}
    if seeds != set(ALL):
        raise ValueError(f"occupancy {arm}: seeds {sorted(seeds, key=int)}")
    return old + new


def occupancy_section(res: Path) -> str:
    out = []
    for arm in (GSD, GSN):
        rows = occupancy_rows(res, arm)
        main = []
        for tag, seeds in (("10", ALL), ("5", OLD)):
            pooled = defaultdict(list)
            for row in rows:
                if row["training_seed"] in seeds:
                    pooled[(row["manifest"], row["condition"])].append(row)
            for (manifest, condition), group in sorted(pooled.items()):
                steps = np.asarray([float(g["policy_steps"]) for g in group])
                w = lambda k: float(np.sum(steps * np.asarray([float(g[k]) for g in group])) / steps.sum())  # noqa: E731
                main.append([tag, manifest, condition] + [pct(w(k)) for k in OCC_KEYS]
                            + [f"{np.median([float(g['kp_applied_median']) for g in group]):.2f}",
                               f"{np.median([float(g['kd_applied_median']) for g in group]):.2f}"])
        ten = [r for r in main if r[0] == "10"]
        five = [r for r in main if r[0] == "5"]
        head = ["seeds", "manifest", "condition"] + OCC_HEAD + ["median Kp (seed median)", "median Kd (seed median)"]
        per_seed = table(["manifest", "condition", "seed"] + OCC_HEAD
                         + [f"{g} {s}" for g in ("Kp", "Ki", "Kd") for s in ("mean", "median", "p5", "p95")],
                         ([r["manifest"], r["condition"], r["training_seed"]] + [pct(r[k]) for k in OCC_KEYS]
                          + [f"{float(r[f'{g}_applied_{s}']):.2f}" for g in ("kp", "ki", "kd")
                             for s in ("mean", "median", "p5", "p95")]
                          for r in sorted(rows, key=lambda r: (r["manifest"], r["condition"], int(r["training_seed"])))))
        out += [f"**{NAME[arm]}, 10 seeds**", table(head, ten),
                details(f"{NAME[arm]}, original 5 seeds (reproduces POSTHOC_RESULTS_3)", table(head, five)),
                details(f"{NAME[arm]}, per seed", per_seed)]
    return "\n\n".join(out)


# ---------------------------------------------------------------- T07 frozen gains

def frozen_section(res: Path) -> str:
    new_gains = meta(res / "frozen_gains_new" / "frozen_gains.json")["gains"]
    new_ep = read_csv(res / "frozen_eval_new" / "episodes.csv")
    out = []
    for arm in (GSD, GSN):
        gdir, edir = FROZEN_OLD[arm]
        gains = dict(meta(P3 / gdir / "frozen_gains.json")["gains"][arm])
        gains.update(new_gains[arm])
        ep = [r for r in read_csv(P3 / edir / "episodes.csv") if r["controller"].startswith(arm)
              and r["controller"] in (arm, f"{arm}_frozen_clean", f"{arm}_frozen_delay")] + \
             [r for r in new_ep if r["controller"] in (arm, f"{arm}_frozen_clean", f"{arm}_frozen_delay")]
        g_rows = [[s + (" (new)" if s in NEW else "")] + [
            f"{gains[s][k]['mean']['kp']:.2f} / {gains[s][k]['mean']['ki']:.3f} / {gains[s][k]['mean']['kd']:.2f}"
            for k in ("clean", "delay")] for s in ALL]
        per = {v: r2.seed_completion(ep, name, r2.DELAY_CELLS) for v, name in
               (("dyn", arm), ("clean", f"{arm}_frozen_clean"), ("delay", f"{arm}_frozen_delay"))}
        for v, p in per.items():
            if set(p) != set(ALL):
                raise ValueError(f"frozen {arm} {v}: seeds {sorted(p, key=int)}")
        c_rows, flags = [], {"5": 0, "10": 0}
        for s in ALL:
            vals = {v: p[s] for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            flag = gap > 0.05
            flags["10"] += flag
            flags["5"] += flag and s in OLD
            c_rows.append([s + (" (new)" if s in NEW else ""), pct(vals["dyn"]), pct(vals["clean"]),
                           pct(vals["delay"]), f"{gap * 100:+.1f}", "**yes**" if flag else ""])
        for tag, seeds in (("mean, 10 seeds", ALL), ("mean, original 5", OLD)):
            vals = {v: float(np.mean([p[s] for s in seeds])) for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            c_rows.append([f"**{tag}**", pct(vals["dyn"]), pct(vals["clean"]), pct(vals["delay"]),
                           f"{gap * 100:+.1f}", ""])
        out += [f"**{NAME[arm]}: frozen gains (mean applied Kp / Ki / Kd on `validation.json`)**",
                table(["seed", "F-clean", "F-delay"], g_rows),
                f"**{NAME[arm]}: completion in the four delay-bearing cells of the ranged held-out set (%)**",
                table(["seed", "dynamic (online gains)", "F-clean", "F-delay", "dynamic − best frozen (pp)",
                       "flag (> 5 pp)"], c_rows),
                f"Seeds where online gains beat the better frozen variant by more than 5 pp: {flags['10']} of 10 "
                f"({flags['5']} of the original 5)."]
    return "\n\n".join(out)


# ---------------------------------------------------------------- W18 cross-track error

def w18_section(res: Path) -> str:
    ep = [r for r in read_csv(res / "traces_heldout" / "episodes.csv") if r["controller"] in ARMS]
    by = defaultdict(list)
    for r in ep:
        by[(r["controller"], r["evaluation_mode"], r["condition"])].append(r)
    rows = []
    for arm in ARMS:
        line = [NAME[arm]]
        for m in MODES:
            for c in CONDITIONS:
                g = by[(arm, m, c)]
                full = len(g) == (600 if arm in LEARNED else 60)  # partial cells (traced paths only) are not shown
                line.append(f"{np.mean([float(x['mean_abs_e_ct_m']) for x in g]) * 1000:.2f}" if full else "–")
        rows.append(line)
    header = ["arm"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS]
    # clean cell: ranking by mean |e_ct| vs E_FA, with seed range for learned arms
    clean = []
    for arm in ARMS:
        g = by[(arm, "stationary", "nominal")]
        ect = np.mean([float(x["mean_abs_e_ct_m"]) for x in g]) * 1000
        rms = np.mean([float(x["rms_e_ct_m"]) for x in g]) * 1000
        mx = np.mean([float(x["max_abs_e_ct_m"]) for x in g]) * 1000
        efa = np.mean([float(x["failure_adjusted_error_m"]) for x in g]) * 1000
        dist = np.mean([float(x["mean_distance_m"]) for x in g]) * 1000
        comp = np.mean([str(x["finished"]).lower() == "true" for x in g])
        if arm in LEARNED:
            per_seed = defaultdict(list)
            for x in g:
                per_seed[x["training_seed"]].append(float(x["mean_abs_e_ct_m"]))
            means = [np.mean(v) * 1000 for v in per_seed.values()]
            spread = f"{min(means):.2f}–{max(means):.2f}"
        else:
            spread = "–"
        clean.append([NAME[arm], ect, rms, mx, dist, efa, pct(comp), spread])
    rank_ect = {r[0]: i + 1 for i, r in enumerate(sorted(clean, key=lambda r: r[1]))}
    rank_efa = {r[0]: i + 1 for i, r in enumerate(sorted(clean, key=lambda r: r[5]))}
    clean_rows = [[r[0], f"{r[1]:.2f}", rank_ect[r[0]], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}",
                   rank_efa[r[0]], r[6], r[7]] for r in sorted(clean, key=lambda r: r[1])]
    return "\n\n".join([
        "e_ct is the signed cross-track error to the continuous reference path (`core/dynamics.py`), logged at every "
        "policy step (50 Hz). Per episode: mean |e_ct|, RMS and max over all policy steps, including failed runs up "
        "to their end. Seed-pooled over ten seeds (600 runs per learned cell, 60 per PID cell). E_FA and mean path "
        "distance are from the same replays.",
        "**Clean cell (stationary nominal), ranked by mean |e_ct|**",
        table(["arm", "mean |e_ct| mm", "rank", "RMS e_ct mm", "max |e_ct| mm", "mean path distance mm", "E_FA mm",
               "E_FA rank", "completion %", "seed range of mean |e_ct| (mm)"], clean_rows),
        "**Mean |e_ct| (mm), every cell**",
        table(header, rows),
    ])


# ---------------------------------------------------------------- W17 data note

def representative_seed(arm: str) -> str:
    """Seed whose ranged held-out clean-cell E_FA is closest to the ten-seed median (reported rows)."""
    rows = [r for f in (P4 / "heldout_ranged" / "episodes.csv",
                        P3 / "g4_heldout_ranged_g1" / "episodes.csv", P3 / "g4_heldout_ranged_g2" / "episodes.csv",
                        ROOT / "artifacts" / "results" / "posthoc2" / "f2_heldout_ranged" / "episodes.csv")
            for r in read_csv(f) if r["controller"] == arm and r["evaluation_mode"] == "stationary"
            and r["condition"] == "nominal" and r["training_seed"] in ALL]
    per = defaultdict(list)
    for r in rows:
        per[r["training_seed"]].append(float(r["failure_adjusted_error_m"]))
    means = {s: np.mean(v) for s, v in per.items()}
    med = np.median(list(means.values()))
    return min(means, key=lambda s: (abs(means[s] - med), int(s)))


def w17_section(res: Path) -> tuple[str, dict[str, str]]:
    reps = {arm: representative_seed(arm) for arm in LEARNED}
    th = read_csv(res / "traces_heldout" / "episodes.csv")
    td = read_csv(res / "traces_dynamic" / "episodes.csv")
    rows = []
    for p in PATHS:
        for arm in ARMS:
            seed = reps.get(arm, "")
            for r in th + td:
                if r["controller"] == arm and r["training_seed"] == seed and f"_{p}-v0.5-" in r["scenario_id"]:
                    traced = r["scenario_id"].endswith(("stationary-nominal-ranged", "stationary-combined-ranged",
                                                        "transient-delay-ranged", "transient-combined-ranged")) \
                        or "dynamic" in r["scenario_id"]
                    if traced:
                        rows.append([p, NAME[arm], seed or "–", r["scenario_id"].split("-v0.5-")[1],
                                     "yes" if str(r["finished"]).lower() == "true" else "**no**",
                                     mm(r["failure_adjusted_error_m"]), mm(r["mean_abs_e_ct_m"])])
    files = table(["file", "content"], [
        ["`path_points.csv`", "x, y, s, curvature of every training, validation and test path (1 point per 5 cm)"],
        ["`path_summary.csv`", "one row per path: split, id, length, max / RMS curvature"],
        ["`traces_heldout.csv.gz`", "per-step traces, ranged test set, paths 10019 / 10001 / 10014 at 0.5 m/s, "
                                    "four scenarios each (stationary nominal, stationary combined, transient delay, "
                                    "transient combined), all 7 controllers × 10 seeds"],
        ["`traces_dynamic.csv.gz`", "per-step traces, dynamic delay, same three paths at 0.5 m/s, D_max 0.15 / "
                                    "0.3 / 0.45 s, delay only and combined, all 7 controllers × 10 seeds"],
        ["`episodes_traced.csv`", "episode metrics (E_FA, completion, mean |e_ct|) of every traced run"],
    ])
    cols = ("controller, training_seed (empty for PIDs), scenario_id, step, t [s], x, y [m], e_ct, e_ct_measured, "
            "distance_m, progress_m, speed_mps, actuator_delay_s, sensor_noise_m, omega_requested, omega_applied, "
            "kp, ki, kd (applied; fixed values for PIDs, empty for DR), kp_target, ki_target, kd_target")
    text = "\n\n".join([
        f"Files in `{THESIS_W17.as_posix()}/` (also in `<results>/traces_*`).", files,
        f"**Trace columns:** {cols}. One row per policy step (0.02 s). Paths start at the origin heading +x; the "
        "robot starts there too, so trajectories and paths share coordinates.",
        "**Paths:** " + "; ".join(f"{p} {PATH_NOTE[p]}" for p in PATHS) + ".",
        "**Representative seed per learned controller** (seed whose clean-cell E_FA on the ranged test set is "
        "closest to the ten-seed median): " + ", ".join(f"{NAME[a]} seed {s}" for a, s in reps.items()) + ".",
        details("Traced runs of the representative seeds", table(
            ["path", "arm", "seed", "scenario", "finished", "E_FA mm", "mean |e_ct| mm"], rows)),
    ])
    return text, reps


def export_w17(res: Path) -> None:
    THESIS_W17.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(res / "traces_heldout" / "traces.csv.gz", THESIS_W17 / "traces_heldout.csv.gz")
    shutil.copyfile(res / "traces_dynamic" / "traces.csv.gz", THESIS_W17 / "traces_dynamic.csv.gz")
    traced = []
    for name in ("traces_heldout", "traces_dynamic"):
        ids = set()
        import gzip
        import csv
        with gzip.open(res / name / "traces.csv.gz", "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ids.add(row["scenario_id"])
        traced += [r for r in read_csv(res / name / "episodes.csv") if r["scenario_id"] in ids]
    from metrics import write_csv
    write_csv(THESIS_W17 / "episodes_traced.csv", traced)
    for f in ("path_points.csv", "path_summary.csv"):
        src = P4 / "w17" / f
        if src.is_file():
            shutil.copyfile(src, THESIS_W17 / f)


# ---------------------------------------------------------------- build

def observations() -> dict[str, str]:
    path = ROOT / "posthoc_observations5.md"
    if not path.is_file():
        return {}
    out, key, lines = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if key:
                out[key] = "\n".join(lines).strip()
            key, lines = line[3:].strip(), []
        elif key:
            lines.append(line)
    if key:
        out[key] = "\n".join(lines).strip()
    return out


def build(res: Path) -> str:
    obs = observations()
    o = lambda key: obs.get(key, "_(observations not written yet)_")  # noqa: E731
    local = res.name.endswith("_local")
    banner = ("> **Internal: provisional laptop run** (`posthoc5_local`). Numbers are close to, but not bit-identical "
              "with, the cluster rows (§0). Replace with the cluster job (`slurm_t02.sh`, job 3366201) before final "
              "submission: rerun `python posthoc_report5.py` once `artifacts/results/posthoc5` exists. Do not mention "
              "this in the thesis text." if local else "")
    w17, _ = w17_section(res)
    out = [
        "# Experiment Results 5",
        "T02 replays: ten-seed gain diagnostics of the schedulers (T07), cross-track error (W18) and trajectory / "
        "gain traces (W17). Replays of existing checkpoints only, no training. Generated by `posthoc_report5.py`. "
        "Arm names are the thesis names (GS-D = GS-D-W 2M, GS-N = GS-N-W).",
        banner,
        "## Summary", o("Summary"),
        "## 0. Run record", run_record(res),
        "## 1. Gain occupancy, ten seeds (T07)",
        "Same definitions as `POSTHOC_RESULTS_3.md` §1.2 / §12.1: edges in action coordinates (|a| ≥ 0.98), the Kp "
        "lower edge is Kp ≤ 1.5 on the wide range; percentages step-weighted over seeds; medians are the median of "
        "the per-seed medians. GS-D on `validation_ranged.json` and the ranged test set, GS-N on `validation.json` "
        "and the ranged test set.",
        occupancy_section(res), o("Occupancy"),
        "## 2. Frozen gains, ten seeds (T07)",
        "Same construction as `POSTHOC_RESULTS_3.md` §8.1 / §12.2: mean applied gains on `validation.json` (F-clean "
        "from nominal scenarios, F-delay from delay and combined scenarios), evaluated as constant gains on the ranged "
        "test set with the checkpoint's own gain range and rate limit.",
        frozen_section(res), o("Frozen"),
        "## 3. Cross-track error (W18)", w18_section(res), o("W18"),
        "## 4. Trace data for the task figures (W17)", w17, o("W17"),
        "## 5. Files",
        f"- **Results:** `{res.relative_to(ROOT).as_posix()}/` (occupancy_*, frozen_*, traces_*).\n"
        "- **Old seeds:** `artifacts/results/posthoc3/` (g5_occupancy_2m, g4_occupancy_wide, g5_frozen_*, "
        "g4_frozen_*).\n- **Code:** `slurm_t02.sh`, `posthoc_traces.py`, `posthoc_path_geometry.py`, "
        "`posthoc_report5.py`.",
    ]
    return "\n\n".join(part for part in out if part) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results")
    parser.add_argument("--output", default=str(ROOT / "POSTHOC_RESULTS_5.md"))
    parser.add_argument("--copy-to", default="D:/Msc/Thesis_Final/Experiments/POSTHOC_RESULTS_5.md")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    cluster = ROOT / "artifacts" / "results" / "posthoc5"
    res = Path(args.results).resolve() if args.results else (
        cluster if (cluster / "traces_dynamic" / "episodes.csv").is_file()
        else ROOT / "artifacts" / "results" / "posthoc5_local")
    text = build(res)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({text.count(chr(10))} lines) from {res.name}")
    if not args.no_export:
        export_w17(res)
        print(f"exported W17 data to {THESIS_W17}")
    if args.copy_to:
        shutil.copyfile(args.output, args.copy_to)
        print(f"copied to {args.copy_to}")


if __name__ == "__main__":
    main()
