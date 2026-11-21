"""Build POSTHOC_RESULTS_4.md (plan 4: five confirmation seeds) from
artifacts/results/posthoc4 plus the five-seed rows of plans 2 and 3.

Plan 4 retrained the four learned arms of the thesis comparison (GS-D = GS-D-W 2M,
GS-N = GS-N-W, DR-D, DR-N) with seeds 83, 97, 109, 127, 149, declared before the
run, every setting frozen at its five-seed value (slurm_plan4.sh, job 3358307).
Nothing is re-selected: the arms, boxes and budgets are the plan-3 outcome.

Every comparison is shown for the original five seeds and for all ten. The
five-seed numbers reproduce POSTHOC_RESULTS_2/3 (same rows, same bootstrap).
Bootstrap results are cached in artifacts/results/posthoc4/paired_cache.json.

    python posthoc_report4.py
    python posthoc_report4.py --workers 12 --copy-to D:/Msc/Thesis_Final/Experiments/POSTHOC_RESULTS_4.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

import posthoc_report as r1
import posthoc_report3 as r3
from posthoc_report import (
    CONDITIONS, MODES, breaking_table, cell, details, full_metric_table, index, levels_of, mm, pct,
    sweep_tables, table, training_cost,
)
from metrics import read_csv

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "artifacts" / "results" / "posthoc2"
P3 = ROOT / "artifacts" / "results" / "posthoc3"
P4 = ROOT / "artifacts" / "results" / "posthoc4"
CACHE = P4 / "paired_cache.json"
JOB_LOG = ROOT / "slurm-thesis-plan4-3358307.out"
PLAN3_LOG = ROOT / "slurm-thesis-plan3-3281395.out"

OLD = tuple(r1.SEEDS)
NEW = ("83", "97", "109", "127", "149")
ALL = OLD + NEW

GSD, GSN = r3.CHOSEN_D, r3.CHOSEN_N
DRD, DRN = r3.DRD, r3.DRN
PIDN, PIDR, PIDRN = "pid_global_nominal", "pid_global_robust", r3.PIDRN
LEARNED = [GSN, GSD, DRN, DRD]
FIXED = [PIDN, PIDR, PIDRN]
ARMS = [PIDN, PIDR, PIDRN, GSN, GSD, DRN, DRD]
# Thesis names: the chosen scheduler versions take the plain names.
NAME = {PIDN: "PID-N", PIDR: "PID-R", PIDRN: "PID-RN", GSN: "GS-N", GSD: "GS-D", DRN: "DR-N", DRD: "DR-D"}
PAIRS = [(GSD, PIDR), (GSD, PIDRN), (GSD, DRD), (DRD, PIDRN), (GSN, PIDN), (DRN, PIDN), (GSN, DRN)]
METRICS = (("finished", 100, "ΔCompletion (pp)", "{:+.1f}"), ("failure_adjusted_error_m", 1000, "ΔE_FA (mm)", "{:+.2f}"))
SET_DIR = {"heldout_ranged": "heldout_ranged", "heldout_fixed": "heldout_fixed", "delay": "delay",
           "dynamic": "dynamic", "noise": "noise"}
SUMMARY_METRICS = ("failure_adjusted_error_m", "mean_distance_m", "max_distance_m", "steer_total_variation_per_s",
                   "wheel_saturation_fraction", "speed_ratio", "gain_total_variation_per_s",
                   "post_event_peak_distance_m", "post_event_excess_iae_m_s")
DISTURBED = [(m, c) for m in MODES for c in ("delay", "noise", "combined")]

for arm, short in NAME.items():
    r1.SHORT[arm] = short


# ---------------------------------------------------------------- data

def episodes(name: str) -> list[dict[str, str]]:
    """Five-seed rows of plans 2/3 for the main arms, plus the plan-4 rows."""
    old = [row for row in r3.load(name, "episodes.csv") if row["controller"] in ARMS
           and (row["controller"] not in LEARNED or row["training_seed"] in OLD)]
    new = read_csv(P4 / SET_DIR[name] / "episodes.csv")
    rows = old + new
    counts = defaultdict(int)
    for row in rows:
        counts[(row["controller"], row["training_seed"])] += 1
    for arm in LEARNED:
        seen = {seed for (a, seed) in counts if a == arm}
        if seen != set(ALL):
            raise ValueError(f"{name}: {arm} has seeds {sorted(seen)}")
        sizes = {counts[(arm, seed)] for seed in ALL}
        if len(sizes) != 1:
            raise ValueError(f"{name}: {arm} has unequal episode counts per seed {sizes}")
    return rows


def summarize(rows: list[dict[str, str]], by_seed: bool) -> list[dict[str, Any]]:
    """posthoc_eval.summarize for CSV rows (finished is the string 'True'/'False')."""
    keys = ["controller", "training_seed", "manifest", "sweep", "level", "level_label", "evaluation_mode", "condition"]
    if not by_seed:
        keys.remove("training_seed")
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        record: dict[str, Any] = dict(zip(keys, key))
        record["episodes"] = len(group)
        record["seeds"] = len({row["training_seed"] for row in group})
        record["completion_rate"] = float(np.mean([r1.finished(row) for row in group]))
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            record[metric] = float(np.mean(values)) if values else ""
        out.append(record)
    return out


def summaries(name: str, ep: list[dict[str, str]], seeds: tuple[str, ...]):
    """(pooled index, per-seed index) for the main arms; learned arms restricted to `seeds`."""
    learned = [row for row in ep if row["controller"] in LEARNED and row["training_seed"] in seeds]
    fixed = [row for row in r3.load(name) if row["controller"] in FIXED]
    pooled = fixed + summarize(learned, by_seed=False)
    return index(pooled), index(summarize(learned, by_seed=True), with_seed=True), levels_of(pooled)


# ---------------------------------------------------------------- paired differences

def _level(row: dict[str, str]) -> float | None:
    return round(float(row["level"]), 7) if row.get("level") else None


def paired_values(ep, a, b, mode, condition, level, seeds, key):
    """Seed -> array of (a - b) per scenario; b matched by seed when it is a learned arm."""
    match = b in LEARNED
    ref = {}
    for row in ep:
        if row["controller"] == b and row["evaluation_mode"] == mode and row["condition"] == condition \
                and _level(row) == level:
            ref[(row["training_seed"] if match else "", row["scenario_id"])] = row
    out = defaultdict(list)
    for row in ep:
        if row["controller"] != a or row["evaluation_mode"] != mode or row["condition"] != condition \
                or _level(row) != level or row["training_seed"] not in seeds:
            continue
        base = ref.get((row["training_seed"] if match else "", row["scenario_id"]))
        if base is None:
            continue
        if key == "finished":
            out[row["training_seed"]].append(r1.finished(row) - r1.finished(base))
        else:
            out[row["training_seed"]].append(float(row[key]) - float(base[key]))
    return {seed: np.asarray(values) for seed, values in out.items()}


def _estimate(task):
    cache_key, values = task
    return cache_key, r1.estimate(values)


def cells_of(ep) -> list[tuple[str, str, float | None]]:
    cells = {(row["evaluation_mode"], row["condition"], _level(row)) for row in ep if row["controller"] in LEARNED}
    order = {m: i for i, m in enumerate(MODES)}
    corder = {c: i for i, c in enumerate(CONDITIONS)}
    return sorted(cells, key=lambda c: (order[c[0]], corder[c[1]], -1 if c[2] is None else c[2]))


def compute_paired(data: dict[str, list[dict[str, str]]], workers: int) -> dict[str, list[float]]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    tasks = []
    for name, ep in data.items():
        arms = {row["controller"] for row in ep}
        for a, b in PAIRS:
            if a not in arms or b not in arms:
                continue
            for mode, condition, level in cells_of(ep):
                for key, *_ in METRICS:
                    for tag, seeds in (("5", OLD), ("10", ALL)):
                        cache_key = f"{name}|{a}|{b}|{mode}|{condition}|{level}|{key}|{tag}"
                        if cache_key in cache:
                            continue
                        values = paired_values(ep, a, b, mode, condition, level, seeds, key)
                        if values:
                            tasks.append((cache_key, values))
    if tasks:
        print(f"bootstrapping {len(tasks)} comparisons on {workers} workers")
        with Pool(workers) as pool:
            for done, (cache_key, result) in enumerate(pool.imap_unordered(_estimate, tasks, chunksize=4), 1):
                cache[cache_key] = list(result)
                if done % 200 == 0:
                    print(f"  {done}/{len(tasks)}")
                    CACHE.write_text(json.dumps(cache), encoding="utf-8")
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def clear(est) -> bool:
    return est[1] > 0 or est[2] < 0


def verdict(e5, e10) -> str:
    if clear(e5) == clear(e10):
        if clear(e10) and np.sign(e5[0]) != np.sign(e10[0]):
            return "**sign flip**"
        return "same" if not clear(e10) else "clear in both"
    return "**now clear**" if clear(e10) else "**no longer clear**"


def paired_rows(cache, name, ep, level_col: bool, only_changed: bool):
    """Rows per metric: pair, mode, condition, [level], 5 seeds, 10 seeds, verdict."""
    arms = {row["controller"] for row in ep}
    out = {key: [] for key, *_ in METRICS}
    for a, b in PAIRS:
        if a not in arms or b not in arms:
            continue
        for mode, condition, level in cells_of(ep):
            for key, scale, _, fmt in METRICS:
                e5 = cache.get(f"{name}|{a}|{b}|{mode}|{condition}|{level}|{key}|5")
                e10 = cache.get(f"{name}|{a}|{b}|{mode}|{condition}|{level}|{key}|10")
                if e5 is None or e10 is None:
                    continue
                if all(abs(v) < 1e-12 for v in (*e5, *e10)):
                    continue
                v = verdict(e5, e10)
                if only_changed and not v.startswith("**"):
                    continue
                show = lambda e: f"{fmt.format(e[0] * scale)} [{fmt.format(e[1] * scale)}, {fmt.format(e[2] * scale)}]"  # noqa: E731
                label = [f"{NAME[a]} − {NAME[b]}", mode, condition]
                if level_col:
                    label.append("–" if level is None else r1.level_text(level, "noise" if name == "noise" else "delay"))
                out[key].append(label + [show(e5), show(e10), v])
    return out


def paired_tables(cache, name, ep, level_col=False, only_changed=False) -> str:
    rows = paired_rows(cache, name, ep, level_col, only_changed)
    header = ["pair", "mode", "condition"] + (["level"] if level_col else []) + [
        "5 seeds: mean [95 % CI]", "10 seeds: mean [95 % CI]", "verdict"]
    parts = []
    for key, _, label, _ in METRICS:
        if rows[key]:
            parts.append(f"**{label}**\n\n{table(header, rows[key])}")
        elif only_changed:
            parts.append(f"**{label}**: no verdict changes.")
    return "\n\n".join(parts)


def change_counts(cache, data) -> str:
    out = []
    totals = defaultdict(int)
    for name, ep in data.items():
        rows = paired_rows(cache, name, ep, True, False)
        counts = defaultdict(int)
        for key in rows:
            for row in rows[key]:
                counts[row[-1]] += 1
                totals[row[-1]] += 1
        n = sum(counts.values())
        out.append([name, n, counts["clear in both"], counts["same"], counts["**now clear**"],
                    counts["**no longer clear**"], counts["**sign flip**"]])
    out.append(["**all**", sum(totals.values()), totals["clear in both"], totals["same"], totals["**now clear**"],
                totals["**no longer clear**"], totals["**sign flip**"]])
    return table(["set", "comparisons", "clear in both", "unclear in both", "now clear", "no longer clear",
                  "clear in both, sign flipped"], out)


# ---------------------------------------------------------------- sections

def run_record() -> str:
    text = JOB_LOG.read_text(encoding="utf-8")
    head = {k: re.search(rf"^{k}\s*: (.*)$", text, re.M).group(1) for k in ("host", "cpu", "job", "commit", "seeds")}
    stages = re.findall(r"^=== (.*) \((\d\d:\d\d:\d\d) UTC\) ===$", text, re.M)
    evals = re.findall(r"(\d+) episodes in ([\d.]+) min -> .*/posthoc4/(\S+)", text)
    notes4 = re.findall(r"note: .*", text)
    notes3 = re.findall(r"note: .*", PLAN3_LOG.read_text(encoding="utf-8")) if PLAN3_LOG.is_file() else []
    parity = ("the same drift notes as plan-3 job 3281395 (pre-existing, within tolerance)"
              if notes4 == notes3 else f"{len(notes4)} drift notes; compare with job 3281395")
    rows = [["host", head["host"]], ["CPU", head["cpu"]], ["job", head["job"]], ["commit", f"`{head['commit']}`"],
            ["new seeds", head["seeds"]], ["parity gate", parity]]
    rows += [[f"stage: {s}", f"{t} UTC"] for s, t in stages]
    eval_table = table(["evaluation", "episodes", "wall min"], ([d, n, m] for n, m, d in evals))
    cost = []
    for arm in [GSD, GSN, DRD, DRN]:
        cost.append(f"**{NAME[arm]}** (`{arm}`)\n\n{training_cost(P4 / f'training_{arm}.log')}")
    return "\n\n".join([table(["item", "value"], rows), eval_table,
                        "All 20 runs trained at once on 64 cores; wall times are not comparable with plans 2 and 3.",
                        details("Training wall times", "\n\n".join(cost))])


def protocol() -> str:
    arms = table(["thesis name", "arm", "first five seeds from", "settings"], [
        ["GS-D", f"`{GSD}` (GS-D-W 2M)", "plan 3, job 3281395", "wide box, H = 1, 2M steps, ranged sampler"],
        ["GS-N", f"`{GSN}` (GS-N-W)", "plan 3, job 3281395", "wide box, H = 1, 300k steps, nominal sampler"],
        ["DR-D", f"`{DRD}`", "plan 2, job 3225586", "H = 1, 2M steps, ranged sampler"],
        ["DR-N", f"`{DRN}`", "plan 2, job 3225586", "H = 1, 2M steps, nominal sampler"],
        ["PID-N, PID-R, PID-RN", "fixed gains", "plan 2", "no training seed; rows reused unchanged"],
    ])
    return "\n\n".join([
        "- **Question:** do the plan-3 conclusions hold with ten training seeds instead of five?",
        "- **New seeds:** 83, 97, 109, 127, 149, written into `slurm_plan4.sh` before the run.\n"
        "- **Frozen:** arms, calibration boxes, rate limits, samplers, budgets, PPO settings, checkpoint rule and "
        "evaluation manifests. Each new run uses the exact `train.py` arguments of its five-seed run; the metadata of "
        "the five-seed runs was checked against the new commands before submission.\n"
        "- **Not redone:** the plan-3 selection (wide box, 2M budget) stays based on seeds 11–71 and validation data "
        "only. The new seeds enter the held-out results, not the choice of arm.\n"
        "- **Statistics:** as in plans 2 and 3. Paired differences per scenario; seed-mean with a hierarchical "
        "bootstrap 95 % CI (seeds, then scenarios within seed; 10,000 draws; `analyze.py` seed). Against a fixed PID "
        "the difference is taken per scenario; between two learned arms it is matched by seed. *Clear* means the CI "
        "excludes 0.\n"
        "- **Five-seed columns** reproduce the plan-2/3 numbers (same rows and bootstrap).",
        "**Arms**", arms,
    ])


def validation_section() -> str:
    old = read_csv(P3 / "g3_validation" / "validation_selected.csv") + \
        read_csv(P2 / "f0_validation" / "validation_selected.csv")
    new = read_csv(P4 / "validation" / "validation_selected.csv")
    by = {}
    for row in old + new:
        if row["controller"] in LEARNED and str(row["seed"]) in ALL:
            by[(row["controller"], str(row["seed"]))] = row
    rows = []
    for arm in [GSD, GSN, DRD, DRN]:
        for tag, seeds in (("5", OLD), ("new 5", NEW), ("10", ALL)):
            got = [by[(arm, s)] for s in seeds]
            comp = np.mean([float(r["completion_rate"]) for r in got])
            err = np.mean([float(r["failure_adjusted_error_m"]) for r in got])
            rows.append([NAME[arm], tag, pct(comp), mm(err)])
    per_seed = table(["arm", "seed", "selected step", "val completion %", "val E_FA mm"],
                     ([NAME[a], s, f"{int(by[(a, s)]['selected_timesteps']):,}", pct(by[(a, s)]["completion_rate"]),
                       mm(by[(a, s)]["failure_adjusted_error_m"])] for a in [GSD, GSN, DRD, DRN] for s in ALL))
    return "\n\n".join([
        "Validation score of the checkpoint each run kept (same rule as before: best completion, then lowest E_FA; "
        "first wins a tie). GS-D and DR-D validate on `validation_ranged.json`, GS-N and DR-N on `validation.json`.",
        table(["arm", "seeds", "val completion % (seed-mean)", "val E_FA mm (seed-mean)"], rows),
        details("Per seed", per_seed),
    ])


def heldout_section(data, cache, name, title) -> str:
    ep = data[name]
    idx5, _, _ = summaries(name, ep, OLD)
    idx10, idx_seed, _ = summaries(name, ep, ALL)
    seed_rows = []
    for arm in [GSD, GSN, DRD, DRN]:
        for s in ALL:
            seed_rows.append([NAME[arm], s + (" (new)" if s in NEW else "")] +
                             [cell(idx_seed.get((arm, None, m, c, s))) for m in MODES for c in CONDITIONS])
    header = ["arm", "seed"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS]
    return "\n\n".join([
        f"**{title}, 10 seeds** (completion % / E_FA mm, seed-pooled; PID rows have no training seed)",
        r1.heldout_table(idx10, ARMS),
        details("Same table, original 5 seeds", r1.heldout_table(idx5, ARMS)),
        details("Per seed, learned arms", table(header, seed_rows)),
        "**Paired differences** (row arm − column arm; positive ΔE_FA = row arm worse, positive ΔCompletion = row "
        "arm better)",
        paired_tables(cache, name, ep),
    ])


def sweep_section(data, cache, name, conditions, kind) -> str:
    ep = data[name]
    idx5, idx_seed, levels = summaries(name, ep, OLD)
    idx10, idx_seed, levels = summaries(name, ep, ALL)
    per_seed = []
    for arm in [GSD, GSN, DRD, DRN]:
        for mode in MODES:
            for condition in conditions:
                if not any(k[0] == arm and k[2] == mode and k[3] == condition for k in idx_seed):
                    continue
                per_seed.append(f"**{NAME[arm]}, {mode} {condition}**\n\n" + table(
                    ["level"] + [s for s in ALL],
                    ([r1.level_text(lv, kind)] + [cell(idx_seed.get((arm, lv, mode, condition, s))) for s in ALL]
                     for lv in levels)))
    return "\n\n".join([
        f"**10 seeds** (completion % / E_FA mm, seed-pooled)",
        sweep_tables(idx10, ARMS, levels, kind, conditions),
        "**Breaking points, 10 seeds** (E1 rule)",
        breaking_table(idx10, ARMS, levels, conditions, kind),
        details("Breaking points, original 5 seeds", breaking_table(idx5, ARMS, levels, conditions, kind)),
        details("Same tables, original 5 seeds", sweep_tables(idx5, ARMS, levels, kind, conditions)),
        details("All metrics, 10 seeds", full_metric_table(idx10, ARMS, levels, kind, conditions)),
        details("Per seed, learned arms", "\n\n".join(per_seed)),
        "**Paired differences whose verdict changed** (5 → 10 seeds)",
        paired_tables(cache, name, ep, level_col=True, only_changed=True),
        details("All paired differences", paired_tables(cache, name, ep, level_col=True)),
    ])


def dynamic_section(data, cache) -> str:
    ep = data["dynamic"]
    idx5, _, levels = summaries("dynamic", ep, OLD)
    idx10, idx_seed, levels = summaries("dynamic", ep, ALL)
    tables = []
    for variant in ("delay", "combined"):
        for tag, idx in (("10 seeds", idx10), ("original 5 seeds", idx5)):
            body = table(["D_max"] + [NAME[a] for a in ARMS],
                         ([f"{lv:g} s"] + [cell(idx.get((a, lv, "stationary", variant))) for a in ARMS]
                          for lv in levels))
            block = f"**{variant}, {tag}** (completion % / E_FA mm, seed-pooled)\n\n{body}"
            tables.append(block if tag == "10 seeds" else details(f"{variant}, {tag}", body))
    return "\n\n".join([
        *tables,
        "**Paired differences**",
        paired_tables(cache, "dynamic", ep, level_col=True),
    ])


def reliability_section(data) -> str:
    """Per seed: worst disturbed completion and mean disturbed E_FA on the ranged held-out set."""
    ep = data["heldout_ranged"]
    _, idx_seed, _ = summaries("heldout_ranged", ep, ALL)
    rows, flags = [], defaultdict(lambda: [0, 0])
    for arm in [GSD, GSN, DRD, DRN]:
        for s in ALL:
            comps = [float(idx_seed[(arm, None, m, c, s)]["completion_rate"]) for m, c in DISTURBED]
            errs = [float(idx_seed[(arm, None, m, c, s)]["failure_adjusted_error_m"]) for m, c in DISTURBED]
            nominal = idx_seed[(arm, None, "stationary", "nominal", s)]
            worst = min(comps)
            low = worst < 0.9
            flags[arm][0 if s in OLD else 1] += low
            rows.append([NAME[arm], s + (" (new)" if s in NEW else ""), pct(nominal["completion_rate"]),
                         mm(nominal["failure_adjusted_error_m"]), pct(worst), mm(float(np.mean(errs))),
                         "**below 90 %**" if low else ""])
    summary = table(["arm", "seeds 11–71 below 90 %", "seeds 83–149 below 90 %", "all 10"],
                    ([NAME[a], flags[a][0], flags[a][1], flags[a][0] + flags[a][1]] for a in [GSD, GSN, DRD, DRN]))
    return "\n\n".join([
        "Ranged held-out set. *Worst disturbed completion* is the lowest completion over the six disturbed cells "
        "(stationary and transient × delay, noise, combined). A seed is flagged when that value is below 90 %.",
        summary,
        table(["arm", "seed", "nominal completion %", "nominal E_FA mm", "worst disturbed completion %",
               "mean disturbed E_FA mm", "flag"], rows),
    ])


def observations() -> dict[str, str]:
    path = ROOT / "posthoc_observations4.md"
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


def build(workers: int) -> str:
    data = {name: episodes(name) for name in SET_DIR}
    cache = compute_paired(data, workers)
    obs = observations()
    o = lambda key: obs.get(key, "_(observations not written yet)_")  # noqa: E731
    out = [
        "# Experiment Results 4",
        "Plan 4: five confirmation seeds for the four learned arms of the thesis comparison. Generated by "
        "`posthoc_report4.py` from `artifacts/results/posthoc4/` and the five-seed rows of plans 2 and 3. "
        "Arm names are the thesis names (GS-D = GS-D-W 2M, GS-N = GS-N-W). Rates are seed-pooled unless marked; "
        "E_FA in mm.",
        "## Summary", o("Summary"),
        "## 0. Protocol", protocol(),
        "## 1. Run record", run_record(),
        "## 2. Validation scores of the kept checkpoints", validation_section(), o("Validation"),
        "## 3. How many verdicts changed",
        "Every paired comparison in §4–§7 (seven pairs × every cell × completion and E_FA), counted by whether its "
        "95 % CI excludes 0 with 5 and with 10 seeds. Comparisons that are exactly 0 in both are left out.",
        change_counts(cache, data), o("Changes"),
        "## 4. Ranged held-out 480",
        heldout_section(data, cache, "heldout_ranged", "Ranged held-out"), o("Ranged held-out"),
        "## 5. Fixed-severity held-out (0.15 s / 0.3 mm)",
        heldout_section(data, cache, "heldout_fixed", "Fixed held-out"), o("Fixed held-out"),
        "## 6. Delay sweep and dynamic delay",
        "### 6.1 Constant-delay sweep 0–0.8 s", sweep_section(data, cache, "delay", ("delay",), "delay"),
        "### 6.2 Dynamic delay", dynamic_section(data, cache), o("Delay"),
        "## 7. Noise sweep (noise only and noise + 0.15 s delay)",
        sweep_section(data, cache, "noise", ("noise", "combined"), "noise"), o("Noise"),
        "## 8. Seed reliability", reliability_section(data), o("Reliability"),
        "## 9. Consequences for the thesis text", o("Thesis"),
        "## 10. Files",
        "- **Results:** `artifacts/results/posthoc4/<set>/` (new seeds only); five-seed rows from "
        "`artifacts/results/posthoc2/` and `posthoc3/`.\n- **Models:** `artifacts/models/posthoc/plan4/`.\n"
        "- **Bootstrap cache:** `artifacts/results/posthoc4/paired_cache.json`.\n"
        "- **Code:** `slurm_plan4.sh`, `posthoc_report4.py`, `--seeds` in `posthoc_eval.py` and "
        "`posthoc_validation_check.py`.\n- **Job log:** `slurm-thesis-plan4-3358307.out`.",
    ]
    return "\n\n".join(part for part in out if part) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "POSTHOC_RESULTS_4.md"))
    parser.add_argument("--copy-to")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    text = build(args.workers)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({text.count(chr(10))} lines)")
    if args.copy_to:
        shutil.copyfile(args.output, args.copy_to)
        print(f"copied to {args.copy_to}")


if __name__ == "__main__":
    main()
