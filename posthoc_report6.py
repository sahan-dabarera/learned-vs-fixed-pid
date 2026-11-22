"""Build POSTHOC_RESULTS_6.md (plan 5: ablation arms and three Ch 6 analyses, seeds 83-149)
from artifacts/results/posthoc6 plus the seed 11-71 rows of the earlier plans.

Plan 5 retrained the ablation arms of Ch 8 with seeds 83, 97, 109, 127, 149, each with the
exact train.py arguments of its seed 11-71 run (slurm_plan5.sh). The Ch 6 analyses (effort,
event metrics, decision time) need no new runs: their columns are in the plan-4 rows.

Each tier can be reported on its own. An arm whose new seeds are missing is shown with
seeds 11-71 only and its ten-seed columns read "pending".
Bootstrap results are cached in artifacts/results/posthoc6/paired_cache.json.

    python posthoc_report6.py
    python posthoc_report6.py --workers 12 --copy-to D:/Msc/Thesis_Final/Experiments/POSTHOC_RESULTS_6.md
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
import posthoc_report2 as r2
import posthoc_report3 as r3
import posthoc_report4 as r4
from posthoc_report import CONDITIONS, MODES, cell, details, index, mm, pct, table, training_cost
from metrics import read_csv

ROOT = Path(__file__).resolve().parent
RES = ROOT / "artifacts" / "results"
PH, P2, P3, P4, P6 = RES / "posthoc", RES / "posthoc2", RES / "posthoc3", RES / "posthoc4", RES / "posthoc6"
CACHE = P6 / "paired_cache.json"
JOB_LOG = RES / "plan5.log"

OLD, NEW, ALL = r4.OLD, r4.NEW, r4.ALL
GSD, GSN, DRD, DRN = r4.GSD, r4.GSN, r4.DRD, r4.DRN          # thesis main arms (GS-D = GS-D-W 2M)
PIDN, PIDR, PIDRN = r4.PIDN, r4.PIDR, r4.PIDRN
MAIN = r4.ARMS
FIXED = r4.FIXED

GSDW = "ppo_scheduler_ranged_h1_wide"
# arm: (thesis name, tier, seed 11-71 source, settings)
ABL = {
    GSDW: ("GS-D-W 300k", 1, "plan 3", "wide box, H = 1, 300k, ranged sampler"),
    "ppo_scheduler_ranged_h1": ("narrow GS-D", 2, "plan 2", "narrow box, H = 1, 300k, ranged sampler"),
    "ppo_scheduler_nominal_h1": ("narrow GS-N", 2, "plan 2", "narrow box, H = 1, 300k, nominal"),
    "ppo_scheduler_ranged_h10_wide": ("GS-D-W H10", 2, "plan 3", "wide box, H = 10, 300k, ranged sampler"),
    "ppo_scheduler_nominal": ("narrow GS-N H10", 2, "sealed", "narrow box, H = 10, 300k, nominal (defaults)"),
    "ppo_scheduler_ranged_h10": ("narrow GS-D H10", 2, "plan 2", "narrow box, H = 10, 300k, ranged sampler"),
    "ppo_scheduler_disturbed": ("narrow GS-D H10 point", 2, "sealed", "narrow box, H = 10, 300k, point sampler"),
    "ppo_scheduler_disturbed_h1": ("narrow GS-D point", 2, "plan 2 (E5)", "narrow box, H = 1, 300k, point sampler"),
    "ppo_direct_nominal": ("DR-N H10", 3, "sealed", "H = 10, 2M, nominal (defaults)"),
    "ppo_direct_ranged_h10": ("DR-D H10", 3, "plan 2", "H = 10, 2M, ranged sampler"),
    "ppo_direct_disturbed": ("DR-D H10 point", 3, "sealed", "H = 10, 2M, point sampler"),
    "ppo_direct_disturbed_h1": ("DR-D point", 3, "plan 2 (E5)", "H = 1, 2M, point sampler"),
}
NAME = dict(r4.NAME)
NAME.update({a: v[0] for a, v in ABL.items()})
for _arm, _name in NAME.items():
    r1.SHORT[_arm] = _name

HISTORY = [  # (one frame, ten frames)
    (DRN, "ppo_direct_nominal"), (DRD, "ppo_direct_ranged_h10"), ("ppo_direct_disturbed_h1", "ppo_direct_disturbed"),
    ("ppo_scheduler_nominal_h1", "ppo_scheduler_nominal"), ("ppo_scheduler_ranged_h1", "ppo_scheduler_ranged_h10"),
    ("ppo_scheduler_disturbed_h1", "ppo_scheduler_disturbed"), (GSDW, "ppo_scheduler_ranged_h10_wide"),
]
RANGE = [("ppo_direct_disturbed_h1", DRD), ("ppo_scheduler_disturbed_h1", "ppo_scheduler_ranged_h1")]  # point, range
BUDGET = [(GSD, GSDW)]
EVENT = [(DRD, PIDRN), (GSD, PIDRN)]
STRESS_ARMS = sorted({a for p in RANGE for a in p})
SETS = {
    "heldout_ranged": sorted(set(MAIN) | set(ABL)),
    "heldout_fixed": sorted(set(MAIN) | set(ABL)),
    "delay": STRESS_ARMS,
    "dynamic": STRESS_ARMS,
}
FR = ("finished", 100, "ΔCompletion (pp)", "{:+.1f}")
EF = ("failure_adjusted_error_m", 1000, "ΔE_FA (mm)", "{:+.2f}")
EVENT_METRICS = (("post_event_peak_distance_m", 1000, "Δpost-event peak distance (mm)", "{:+.2f}"),
                 ("post_event_excess_iae_m_s", 1000, "Δpost-event excess IAE (mm·s)", "{:+.2f}"))
STRESS_DELAY, STRESS_DYN = 0.3, 0.45


# ---------------------------------------------------------------- data

def new_rows(name: str) -> list[dict[str, str]]:
    rows = read_csv(P4 / name / "episodes.csv") if (P4 / name / "episodes.csv").is_file() else []
    for tier in (1, 2, 3):
        path = P6 / f"t{tier}_{name}" / "episodes.csv"
        if path.is_file():
            rows += read_csv(path)
    return rows


def load(name: str) -> tuple[list[dict[str, str]], dict[str, tuple[str, ...]]]:
    """All rows of one set for the arms it needs, and the seeds each learned arm has (OLD or ALL)."""
    keep = set(SETS[name]) | set(FIXED)
    rows = [r for r in r3.load(name, "episodes.csv") if r["controller"] in keep
            and (r["controller"] in FIXED or r["training_seed"] in OLD)]
    rows += [r for r in new_rows(name) if r["controller"] in keep and r["controller"] not in FIXED]
    counts = defaultdict(int)
    for r in rows:
        counts[(r["controller"], r["training_seed"])] += 1
    seeds = {}
    for arm in SETS[name]:
        if arm in FIXED:
            continue
        have = {s for (a, s) in counts if a == arm}
        if not set(OLD) <= have:
            raise ValueError(f"{name}: {arm} misses seeds 11-71 ({sorted(have)})")
        if have not in (set(OLD), set(ALL)):
            raise ValueError(f"{name}: {arm} has a partial new-seed set {sorted(have, key=int)}")
        if len({counts[(arm, s)] for s in have}) != 1:
            raise ValueError(f"{name}: {arm} has unequal episode counts per seed")
        seeds[arm] = ALL if have == set(ALL) else OLD
    return rows, seeds


def level(row) -> float | None:
    return round(float(row["level"]), 7) if row.get("level") else None


def group(rows) -> dict[tuple, list[dict[str, str]]]:
    out = defaultdict(list)
    for r in rows:
        out[(r["controller"], r["evaluation_mode"], r["condition"], level(r))].append(r)
    return out


def summary_index(rows, arms, seeds, by_seed=False):
    sel = [r for r in rows if r["controller"] in arms and (r["controller"] in FIXED or r["training_seed"] in seeds[r["controller"]])]
    return index(r4.summarize(sel, by_seed), with_seed=by_seed)


def tag_of(arms, seeds) -> str:
    return "10 seeds" if all(seeds.get(a, ALL) == ALL for a in arms if a not in FIXED) else "5 seeds"


# ---------------------------------------------------------------- paired bootstrap

def paired_values(grouped, a, b, mode, condition, lv, seeds, key):
    match = b not in FIXED
    ref = {(r["training_seed"] if match else "", r["scenario_id"]): r for r in grouped.get((b, mode, condition, lv), [])}
    out = defaultdict(list)
    for r in grouped.get((a, mode, condition, lv), []):
        if r["training_seed"] not in seeds:
            continue
        base = ref.get((r["training_seed"] if match else "", r["scenario_id"]))
        if base is None:
            continue
        if key == "finished":
            out[r["training_seed"]].append(r1.finished(r) - r1.finished(base))
        elif r.get(key) not in ("", None) and base.get(key) not in ("", None):
            out[r["training_seed"]].append(float(r[key]) - float(base[key]))
    return {s: np.asarray(v) for s, v in out.items() if len(v)}


def _estimate(task):
    cache_key, values = task
    return cache_key, r1.estimate(values)


def cells(grouped, arm, only=None) -> list[tuple[str, str, float | None]]:
    got = {(m, c, lv) for (a, m, c, lv) in grouped if a == arm}
    if only:
        got = {x for x in got if only(*x)}
    mo = {m: i for i, m in enumerate(MODES)}
    co = {c: i for i, c in enumerate(CONDITIONS)}
    return sorted(got, key=lambda x: (mo[x[0]], co[x[1]], -1 if x[2] is None else x[2]))


# Comparison families: (family, set, pairs, metrics, cell filter)
FAMILIES = [
    ("event", "heldout_ranged", EVENT, EVENT_METRICS, lambda m, c, lv: m == "transient" and c != "nominal"),
    ("history", "heldout_ranged", HISTORY, (FR, EF), None),
    ("range", "heldout_ranged", RANGE, (FR, EF), None),
    ("range", "heldout_fixed", RANGE, (FR, EF), None),
    ("range", "delay", RANGE, (FR, EF), None),
    ("range", "dynamic", RANGE, (FR, EF), None),
    ("budget", "heldout_ranged", BUDGET, (FR, EF), None),
    ("budget", "heldout_fixed", BUDGET, (FR, EF), None),
]


def compute(data, workers) -> dict[str, list[float]]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    tasks = []
    for _, name, pairs, metrics, only in FAMILIES:
        grouped, seeds = data[name]["grouped"], data[name]["seeds"]
        for a, b in pairs:
            ten = all(seeds.get(x, ALL) == ALL for x in (a, b) if x not in FIXED)
            for mode, condition, lv in cells(grouped, a, only):
                for key, *_ in metrics:
                    for tag, use in (("5", OLD), ("10", ALL)):
                        if tag == "10" and not ten:
                            continue
                        ck = f"{name}|{a}|{b}|{mode}|{condition}|{lv}|{key}|{tag}"
                        if ck in cache:
                            continue
                        values = paired_values(grouped, a, b, mode, condition, lv, use, key)
                        if values:
                            tasks.append((ck, values))
    if tasks:
        print(f"bootstrapping {len(tasks)} comparisons on {workers} workers")
        with Pool(workers) as pool:
            for done, (ck, result) in enumerate(pool.imap_unordered(_estimate, tasks, chunksize=4), 1):
                cache[ck] = list(result)
                if done % 200 == 0:
                    print(f"  {done}/{len(tasks)}")
                    CACHE.write_text(json.dumps(cache), encoding="utf-8")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def show(e, scale, fmt) -> str:
    return f"{fmt.format(e[0] * scale)} [{fmt.format(e[1] * scale)}, {fmt.format(e[2] * scale)}]"


def verdict(e5, e10) -> str:
    if e10 is None:
        return "pending"
    return r4.verdict(e5, e10)


def paired_table(cache, data, name, pairs, metrics, only=None, level_col=False, pick=None, label=None) -> str:
    grouped, seeds = data[name]["grouped"], data[name]["seeds"]
    parts = []
    for key, scale, title, fmt in metrics:
        rows = []
        for a, b in pairs:
            for mode, condition, lv in cells(grouped, a, only):
                if pick and not pick(mode, condition, lv):
                    continue
                e5 = cache.get(f"{name}|{a}|{b}|{mode}|{condition}|{lv}|{key}|5")
                e10 = cache.get(f"{name}|{a}|{b}|{mode}|{condition}|{lv}|{key}|10")
                if e5 is None:
                    continue
                row = [(label or (lambda x, y: f"{NAME[x]} − {NAME[y]}"))(a, b), mode, condition]
                if level_col:
                    row.append("–" if lv is None else r1.level_text(lv, "delay"))
                row += [show(e5, scale, fmt), show(e10, scale, fmt) if e10 else "pending", verdict(e5, e10)]
                rows.append(row)
        head = ["pair", "mode", "condition"] + (["level"] if level_col else []) + [
            "seeds 11–71: mean [95 % CI]", "10 seeds: mean [95 % CI]", "verdict"]
        if rows:
            parts.append(f"**{title}**\n\n{table(head, rows)}")
    return "\n\n".join(parts)


def per_seed_means(data, name, a, b, key, scale, cell_list) -> str:
    """Seed-mean paired difference per seed for each cell (shows which seeds drive an interval)."""
    grouped, seeds = data[name]["grouped"], data[name]["seeds"]
    use = ALL if all(seeds.get(x, ALL) == ALL for x in (a, b) if x not in FIXED) else OLD
    rows = []
    for mode, condition, lv in cell_list:
        v = paired_values(grouped, a, b, mode, condition, lv, use, key)
        rows.append([f"{mode} {condition}" + ("" if lv is None else f" {lv:g}")]
                    + [f"{np.mean(v[s]) * scale:+.1f}" if s in v else "–" for s in use])
    return table(["cell"] + [s + (" (new)" if s in NEW else "") for s in use], rows)


# ---------------------------------------------------------------- run record

def run_record() -> str:
    if not JOB_LOG.is_file():
        return "_(plan5.log not found)_"
    text = JOB_LOG.read_text(encoding="utf-8", errors="replace")
    head = {k: (re.search(rf"^{k}\s*: (.*)$", text, re.M) or [None, "?"])[1] for k in ("host", "job", "commit", "seeds")}
    cpu = "?"
    for path in sorted(P6.glob("*/metadata.json")):
        cpu = json.loads(path.read_text(encoding="utf-8")).get("machine", {}).get("processor") or cpu
        break
    stages = re.findall(r"^=== (.*) \((\d\d:\d\d:\d\d) UTC\) ===$", text, re.M)
    evals = re.findall(r"(\d+) episodes in ([\d.]+) min -> .*?posthoc6[/\\](\S+)", text)
    rows = [["host", head["host"]], ["CPU", cpu], ["job", head["job"]], ["commit", f"`{head['commit']}`"],
            ["new seeds; tiers", head["seeds"]]]
    rows += [[f"stage: {s}", f"{t} UTC"] for s, t in stages]
    cost = []
    for arm in ABL:
        log = P6 / f"training_{arm}.log"
        if log.is_file():
            cost.append(f"**{NAME[arm]}** (`{arm}`)\n\n{training_cost(log)}")
    return "\n\n".join([table(["item", "value"], rows),
                        table(["evaluation", "episodes", "wall min"], ([d, n, m] for n, m, d in evals)),
                        "The five seeds of one arm trained in parallel (5 workers); arms trained one after another.",
                        details("Training wall times", "\n\n".join(cost) or "none yet")])


def protocol(data) -> str:
    seeds = data["heldout_ranged"]["seeds"]
    rows = [[v[0], f"`{a}`", v[1], v[2], v[3], "10" if seeds[a] == ALL else "**5 (pending)**"] for a, v in ABL.items()]
    return "\n\n".join([
        "- **Question:** do the Ch 8 ablation findings and the Ch 6 effort, event and run-time numbers hold with "
        "ten training seeds?\n"
        "- **New seeds:** 83, 97, 109, 127, 149 (the plan-4 seeds). Seeds 11–71 rows are reused unchanged.\n"
        "- **Frozen:** each arm uses the exact `train.py` arguments of its seed 11–71 run; only the seed changes. "
        "Same checkpoint rule (best validation completion, then lowest E_FA), test manifests and evaluation tools. "
        "The sealed ten-frame arms are retrained with default arguments; every option added since the sealed "
        "commit defaults to the sealed behaviour.\n"
        "- **Tier 1** needs one retraining: GS-D-W 300k. The plan-4 GS-D-W 2M runs kept only their best and final "
        "checkpoints, so no ≤300k model existed. Effort, event metrics and decision time come from the plan-4 "
        "rows (no new runs).\n"
        "- **Statistics:** as in plans 2–4. Paired differences per scenario; seed-mean with a hierarchical "
        "bootstrap 95 % CI (seeds, then scenarios within seed; 10,000 draws). Learned pairs are matched by seed. "
        "*Clear* means the CI excludes 0.",
        "**Arms**",
        table(["name", "arm", "tier", "seeds 11–71 from", "settings", "seeds in this report"], rows),
    ])


# ---------------------------------------------------------------- Tier 1: Ch 6 analyses

def effort_section(data) -> str:
    rows, seeds = data["heldout_ranged"]["rows"], data["heldout_ranged"]["seeds"]
    out = []
    for tag, use in (("10 seeds", ALL), ("seeds 11–71", OLD)):
        idx = summary_index(rows, MAIN, defaultdict(lambda: use))
        for key, title, fmt in (("steer_total_variation_per_s", "Steering total variation (rad/s)", "{:.3f}"),
                                ("wheel_saturation_fraction", "Wheel saturation (% of steps)", "pct"),
                                ("speed_ratio", "Speed ratio (mean speed / target)", "{:.3f}")):
            body = table(["arm"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS], (
                [NAME[a]] + [("–" if (r := idx.get((a, None, m, c))) is None or r.get(key) in ("", None)
                              else (pct(r[key]) if fmt == "pct" else fmt.format(float(r[key]))))
                             for m in MODES for c in CONDITIONS] for a in MAIN))
            block = f"**{title}, {tag}** (ranged held-out, seed-pooled)\n\n{body}"
            out.append(block if tag == "10 seeds" else details(f"{title}, {tag}", body))
    return "\n\n".join(out)


def event_section(data, cache) -> str:
    rows = data["heldout_ranged"]["rows"]
    out = []
    for tag, use in (("10 seeds", ALL), ("seeds 11–71", OLD)):
        body = []
        for a in MAIN:
            line = [NAME[a]]
            for key, scale, *_ in EVENT_METRICS:
                for c in ("delay", "noise", "combined"):
                    vals = [float(r[key]) for r in rows if r["controller"] == a and r["evaluation_mode"] == "transient"
                            and r["condition"] == c and r.get(key) not in ("", None)
                            and (a in FIXED or r["training_seed"] in use)]
                    line.append(f"{np.mean(vals) * scale:.1f}" if vals else "–")
            body.append(line)
        head = ["arm"] + [f"{t} {c}" for t in ("peak (mm)", "excess IAE (mm·s)") for c in ("delay", "noise", "combined")]
        block = f"**Event metrics, {tag}** (transient cells, ranged held-out, seed-pooled means over episodes that define them)\n\n{table(head, body)}"
        out.append(block if tag == "10 seeds" else details(f"Event metrics, {tag}", table(head, body)))
    out += ["**Paired differences against PID-RN** (learned seeds against the single PID instance)",
            paired_table(cache, data, "heldout_ranged", EVENT, EVENT_METRICS,
                         only=lambda m, c, lv: m == "transient" and c != "nominal")]
    return "\n\n".join(out)


def decision_time_section(data) -> str:
    rows = data["heldout_ranged"]["rows"]
    out = []
    for a in MAIN:
        line = [NAME[a]]
        for use in (OLD, ALL):
            r = [x for x in rows if x["controller"] == a and (a in FIXED or x["training_seed"] in use)
                 and x.get("mean_inference_ms") not in ("", None)]
            means = np.asarray([float(x["mean_inference_ms"]) for x in r])
            dec = np.asarray([float(x["policy_decisions"]) for x in r])
            p99 = np.asarray([float(x["p99_inference_ms"]) for x in r])
            line += [len(r), f"{np.sum(means * dec) / np.sum(dec) * 1000:.1f}", f"{np.median(p99) * 1000:.1f}",
                     f"{np.percentile(p99, 95) * 1000:.1f}"]
        out.append(line)
    head = ["arm"] + [f"{t} ({s})" for s in ("11–71", "10 seeds")
                      for t in ("episodes", "mean µs", "median p99 µs", "95th pct p99 µs")]
    return "\n\n".join([
        "Wall time of one `predictor(observation)` call on the ranged held-out run (same definition as "
        "POSTHOC_RESULTS_3 §9.3). Seeds 11–71 and 83–149 were both measured on the cluster's EPYC nodes "
        "(plans 3 and 4, one torch thread per worker); the laptop runs of plan 5 do not enter this table.",
        table(head, out),
    ])


# ---------------------------------------------------------------- held-out tables of the ablation arms

def heldout_section(data, name, arms, title) -> str:
    rows, seeds = data[name]["rows"], data[name]["seeds"]
    idx = summary_index(rows, arms, seeds)
    idx5 = summary_index(rows, arms, defaultdict(lambda: OLD))
    idx_seed = summary_index(rows, arms, seeds, by_seed=True)
    head = ["arm", "seeds"] + [f"{m[:4]}. {c}" for m in MODES for c in CONDITIONS]
    main = table(head, ([NAME[a], "10" if seeds[a] == ALL else "**5**"]
                        + [cell(idx.get((a, None, m, c))) for m in MODES for c in CONDITIONS] for a in arms))
    seed_rows = [[NAME[a], s + (" (new)" if s in NEW else "")]
                 + [cell(idx_seed.get((a, None, m, c, s))) for m in MODES for c in CONDITIONS]
                 for a in arms for s in seeds[a]]
    return "\n\n".join([
        f"**{title}** (completion % / E_FA mm, seed-pooled)", main,
        details(f"{title}, seeds 11–71", r1.heldout_table(idx5, arms)),
        details(f"{title}, per seed", table(["arm", "seed"] + head[2:], seed_rows)),
    ])


# ---------------------------------------------------------------- occupancy and frozen gains

OCC = [  # arm, seed 11-71 dir, new dir
    (GSDW, P3 / "g4_occupancy_wide", P6 / "t1_occupancy"),
    ("ppo_scheduler_ranged_h1", P3 / "g0_occupancy", P6 / "t2_occupancy"),
    ("ppo_scheduler_nominal_h1", P3 / "g0_occupancy", P6 / "t2_occupancy"),
]


def occupancy_section() -> str:
    out = []
    for arm, old_dir, new_dir in OCC:
        rows = [r for r in read_csv(old_dir / "occupancy.csv") if r["controller"] == arm and r["training_seed"] in OLD]
        if (new_dir / "occupancy.csv").is_file():
            rows += [r for r in read_csv(new_dir / "occupancy.csv") if r["controller"] == arm]
        have = {r["training_seed"] for r in rows}
        use = ALL if have == set(ALL) else OLD
        main = []
        for tag, seeds in (("10", ALL), ("5", OLD)):
            if tag == "10" and use != ALL:
                continue
            pooled = defaultdict(list)
            for r in rows:
                if r["training_seed"] in seeds:
                    pooled[(r["manifest"], r["condition"])].append(r)
            for (manifest, condition), g in sorted(pooled.items()):
                steps = np.asarray([float(x["policy_steps"]) for x in g])
                w = lambda k: float(np.sum(steps * np.asarray([float(x[k]) for x in g])) / steps.sum())  # noqa: E731
                main.append([tag, manifest, condition] + [pct(w(k)) for k in OCC_KEYS]
                            + [f"{np.median([float(x['kp_applied_median']) for x in g]):.2f}",
                               f"{np.median([float(x['kd_applied_median']) for x in g]):.2f}"])
        head = ["seeds", "manifest", "condition"] + OCC_HEAD + ["median Kp (seed median)", "median Kd (seed median)"]
        per_seed = table(["manifest", "condition", "seed"] + OCC_HEAD,
                         ([r["manifest"], r["condition"], r["training_seed"]] + [pct(r[k]) for k in OCC_KEYS]
                          for r in sorted(rows, key=lambda r: (r["manifest"], r["condition"], int(r["training_seed"])))))
        ten = [r for r in main if r[0] == "10"]
        five = [r for r in main if r[0] == "5"]
        out += [f"**{NAME[arm]}, {'10 seeds' if ten else 'seeds 11–71 (new seeds pending)'}**",
                table(head, ten or five)]
        if ten:
            out.append(details(f"{NAME[arm]}, seeds 11–71", table(head, five)))
        out.append(details(f"{NAME[arm]}, per seed", per_seed))
    return "\n\n".join(out)


OCC_KEYS = ["kp_target_at_lower_edge", "kp_applied_on_lower_bound", "kd_target_at_lower_edge",
            "kd_applied_on_lower_bound", "ki_target_at_lower_edge", "kp_target_at_upper_edge"]
OCC_HEAD = ["Kp target at lower edge %", "Kp applied on lower bound %", "Kd target at lower edge %",
            "Kd applied on lower bound %", "Ki target at lower edge %", "Kp target at upper edge %"]


FROZEN = [  # arm, old gains json, old eval dir, online source for seeds 11-71 (None: in eval dir), new gains, new eval
    (GSDW, P3 / "g4_frozen_gains", P3 / "g4_frozen_eval", None, P6 / "t1_frozen_gains", P6 / "t1_frozen_eval",
     "ranged held-out"),
    ("ppo_scheduler_ranged_h1", P2 / "f6_frozen_gains", P2 / "f6_frozen_eval", None, P6 / "t2_frozen_gains",
     P6 / "t2_frozen_eval_ranged", "ranged held-out"),
    ("ppo_scheduler_nominal_h1", P2 / "f6_frozen_gains", P2 / "f6_frozen_eval", None, P6 / "t2_frozen_gains",
     P6 / "t2_frozen_eval_ranged", "ranged held-out"),
    ("ppo_scheduler_nominal", PH / "e3_frozen_gains", PH / "e3_frozen_eval", "heldout_fixed", P6 / "t2_frozen_gains",
     P6 / "t2_frozen_eval_fixed", "fixed held-out"),
    ("ppo_scheduler_disturbed", PH / "e3_frozen_gains", PH / "e3_frozen_eval", "heldout_fixed",
     P6 / "t2_frozen_gains", P6 / "t2_frozen_eval_fixed", "fixed held-out"),
]


def frozen_section(data) -> str:
    out = []
    for arm, og, oe, online, ng, ne, label in FROZEN:
        names = (arm, f"{arm}_frozen_clean", f"{arm}_frozen_delay")
        gains = dict(json.loads((og / "frozen_gains.json").read_text(encoding="utf-8"))["gains"][arm])
        ep = [r for r in read_csv(oe / "episodes.csv") if r["controller"] in names and r["training_seed"] in OLD]
        if online:
            ep += [r for r in data[online]["rows"] if r["controller"] == arm and r["training_seed"] in OLD]
        if (ng / "frozen_gains.json").is_file() and (ne / "episodes.csv").is_file():
            gains.update(json.loads((ng / "frozen_gains.json").read_text(encoding="utf-8"))["gains"].get(arm, {}))
            ep += [r for r in read_csv(ne / "episodes.csv") if r["controller"] in names]
        per = {v: r2.seed_completion(ep, n, r2.DELAY_CELLS) for v, n in zip(("dyn", "clean", "delay"), names)}
        use = ALL if all(set(p) == set(ALL) for p in per.values()) else OLD
        for v, p in per.items():
            if not set(OLD) <= set(p):
                raise ValueError(f"frozen {arm} {v}: seeds {sorted(p, key=int)}")
        g_rows = [[s + (" (new)" if s in NEW else "")] + [
            f"{gains[s][k]['mean']['kp']:.2f} / {gains[s][k]['mean']['ki']:.3f} / {gains[s][k]['mean']['kd']:.2f}"
            for k in ("clean", "delay")] for s in use]
        c_rows, flags = [], {"5": 0, "10": 0}
        for s in use:
            vals = {v: p[s] for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            flags["10"] += gap > 0.05
            flags["5"] += gap > 0.05 and s in OLD
            c_rows.append([s + (" (new)" if s in NEW else ""), pct(vals["dyn"]), pct(vals["clean"]),
                           pct(vals["delay"]), f"{gap * 100:+.1f}", "**yes**" if gap > 0.05 else ""])
        for tag, seeds in ((("mean, 10 seeds", ALL),) if use == ALL else ()) + (("mean, seeds 11–71", OLD),):
            vals = {v: float(np.mean([p[s] for s in seeds])) for v, p in per.items()}
            gap = vals["dyn"] - max(vals["clean"], vals["delay"])
            c_rows.append([f"**{tag}**", pct(vals["dyn"]), pct(vals["clean"]), pct(vals["delay"]),
                           f"{gap * 100:+.1f}", ""])
        n = "10" if use == ALL else "5"
        out += [f"**{NAME[arm]}** (`{arm}`; frozen gains = mean applied Kp / Ki / Kd on its validation set; "
                f"evaluated on the {label} set)",
                details("Frozen gains per seed", table(["seed", "F-clean", "F-delay"], g_rows)),
                table(["seed", "online gains", "F-clean", "F-delay", "online − best frozen (pp)", "> 5 pp"], c_rows),
                f"Seeds where online gains beat the better frozen variant by more than 5 pp: {flags[n if n == '10' else '5']} "
                f"of {n}" + (f" ({flags['5']} of seeds 11–71)." if n == "10" else ".")]
    return "\n\n".join(out)


# ---------------------------------------------------------------- history, range, budget

def validation_rows() -> dict[tuple[str, str], dict[str, str]]:
    by = {}
    for path in [P2 / "f0_validation", P3 / "g3_validation"]:
        for r in read_csv(path / "validation_selected.csv"):
            if str(r["seed"]) in OLD:
                by[(r["controller"], str(r["seed"]))] = r
    for path in [P4 / "validation"] + [P6 / f"t{t}_validation" for t in (1, 2, 3)]:
        if (path / "validation_selected.csv").is_file():
            for r in read_csv(path / "validation_selected.csv"):
                by[(r["controller"], str(r["seed"]))] = r
    return by


def history_section(data, cache) -> str:
    rows, seeds = data["heldout_ranged"]["rows"], data["heldout_ranged"]["seeds"]
    val = validation_rows()
    agree = []
    for one, ten in HISTORY:
        for tag, use in (("10", ALL), ("5", OLD)):
            if tag == "10" and (seeds[one] != ALL or seeds[ten] != ALL):
                agree.append([f"{NAME[one]} vs {NAME[ten]}", tag, "pending", "", "", "", ""])
                continue
            v = {a: (np.mean([float(val[(a, s)]["completion_rate"]) for s in use]),
                     np.mean([float(val[(a, s)]["failure_adjusted_error_m"]) for s in use])) for a in (one, ten)}
            idx = summary_index(rows, [one, ten], defaultdict(lambda: use))
            test = {a: np.mean([float(idx[(a, None, m, c)]["failure_adjusted_error_m"]) for m in MODES
                                for c in CONDITIONS]) for a in (one, ten)}
            vfav = "H1" if v[one][1] < v[ten][1] else "H10"
            tfav = "H1" if test[one] < test[ten] else "H10"
            agree.append([f"{NAME[one]} vs {NAME[ten]}", tag, f"{pct(v[one][0])} / {mm(v[one][1])}",
                          f"{pct(v[ten][0])} / {mm(v[ten][1])}", vfav, tfav, "yes" if vfav == tfav else "**no**"])
    seed_blocks = []
    for one, ten in HISTORY:
        cl = cells(data["heldout_ranged"]["grouped"], one)
        seed_blocks.append(f"**{NAME[one]} − {NAME[ten]}, ΔE_FA per seed (mm)**\n\n"
                           + per_seed_means(data, "heldout_ranged", one, ten, "failure_adjusted_error_m", 1000, cl))
    return "\n\n".join([
        "One frame minus ten frames, matched by seed, ranged held-out set (the Ch 8 history forest). Negative "
        "ΔE_FA and positive ΔCompletion favour one frame.",
        paired_table(cache, data, "heldout_ranged", HISTORY, (EF, FR)),
        details("Per-seed paired means", "\n\n".join(seed_blocks)),
        "**Validation agreement** (seed-mean validation completion % / E_FA mm of the kept checkpoints; test = "
        "mean E_FA over the eight ranged held-out cells, seed-pooled)",
        table(["pair", "seeds", "one frame (val)", "ten frames (val)", "val favours", "test favours", "agree"], agree),
    ])


def range_values(data) -> str:
    hr, dl, dy = data["heldout_ranged"], data["delay"], data["dynamic"]
    out = []
    for tag, fn in (("10 seeds", lambda d: d["seeds"]), ("seeds 11–71", lambda d: defaultdict(lambda: OLD))):
        arms = [a for p in RANGE for a in p]
        i_h = summary_index(hr["rows"], arms, fn(hr))
        i_d = summary_index(dl["rows"], arms, fn(dl))
        i_y = summary_index(dy["rows"], arms, fn(dy))
        body = []
        for a in arms:
            h = lambda c: i_h.get((a, None, "stationary", c))  # noqa: E731
            stress = i_d.get((a, STRESS_DELAY, "stationary", "delay"))
            dyn = i_y.get((a, STRESS_DYN, "stationary", "delay"))
            dync = i_y.get((a, STRESS_DYN, "stationary", "combined"))
            n = "10" if all(d["seeds"][a] == ALL for d in (hr, dl, dy)) else "5"
            body.append([NAME[a], n if tag == "10 seeds" else "5", mm(h("nominal")["failure_adjusted_error_m"])]
                        + [x for c in ("delay", "noise", "combined")
                           for x in (pct(h(c)["completion_rate"]), mm(h(c)["failure_adjusted_error_m"]))]
                        + [pct(stress["completion_rate"]) if stress else "–", pct(dyn["completion_rate"]) if dyn else "–",
                           pct(dync["completion_rate"]) if dync else "–"])
        head = ["controller", "seeds", "clean E_FA", "delay C", "delay E_FA", "noise C", "noise E_FA", "comb. C",
                "comb. E_FA", f"stress {STRESS_DELAY:g} s C", f"dynamic 0–{STRESS_DYN:g} s C (delay)",
                f"dynamic 0–{STRESS_DYN:g} s C (combined)"]
        block = (f"**Point- vs range-trained, {tag}** (tab:diag_range layout; ranged held-out stationary cells; stress: "
                 f"stationary constant delay and dynamic delay)\n\n{table(head, body)}")
        out.append(block if tag == "10 seeds" else details(f"Same table, {tag}", table(head, body)))
    return "\n\n".join(out)


def range_section(data, cache) -> str:
    lbl = lambda a, b: f"{NAME[a]} − {NAME[b].replace('narrow GS-D', 'narrow GS-D range').replace('DR-D', 'DR-D range')}"  # noqa: E731
    return "\n\n".join([
        range_values(data),
        "**Paired differences, point − range** (matched by seed). Ranged held-out, stationary cells:",
        paired_table(cache, data, "heldout_ranged", RANGE, (FR, EF), pick=lambda m, c, lv: m == "stationary", label=lbl),
        f"**Stress cells** (constant delay {STRESS_DELAY:g} s; dynamic D_max {STRESS_DYN:g} s):",
        paired_table(cache, data, "delay", RANGE, (FR, EF), level_col=True,
                     pick=lambda m, c, lv: m == "stationary" and lv == STRESS_DELAY, label=lbl),
        paired_table(cache, data, "dynamic", RANGE, (FR, EF), level_col=True,
                     pick=lambda m, c, lv: lv == STRESS_DYN, label=lbl),
        details("All cells: ranged held-out (transient too), fixed held-out, every delay level and D_max",
                "\n\n".join([
                    "**Ranged held-out**", paired_table(cache, data, "heldout_ranged", RANGE, (FR, EF), label=lbl),
                    "**Fixed held-out**", paired_table(cache, data, "heldout_fixed", RANGE, (FR, EF), label=lbl),
                    "**Delay sweep**", paired_table(cache, data, "delay", RANGE, (FR, EF), level_col=True, label=lbl),
                    "**Dynamic delay**", paired_table(cache, data, "dynamic", RANGE, (FR, EF), level_col=True,
                                                      label=lbl)])),
    ])


def budget_section(data, cache) -> str:
    return "\n\n".join([
        "GS-D-W 2M (the thesis GS-D) minus GS-D-W 300k, matched by seed. The 2M runs are plans 3 and 4 (cluster); "
        "the 300k runs are plan 3 (seeds 11–71) and plan 5 (seeds 83–149).",
        "**Ranged held-out**", paired_table(cache, data, "heldout_ranged", BUDGET, (FR, EF)),
        "**Fixed held-out**", paired_table(cache, data, "heldout_fixed", BUDGET, (FR, EF)),
    ])


# ---------------------------------------------------------------- build

def observations() -> dict[str, str]:
    path = ROOT / "posthoc_observations6.md"
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
    data = {}
    for name in SETS:
        rows, seeds = load(name)
        data[name] = {"rows": rows, "seeds": seeds, "grouped": group(rows)}
        print(f"{name}: {len(rows)} rows; ten seeds: {sorted(NAME[a] for a, s in seeds.items() if s == ALL)}")
    cache = compute(data, workers)
    obs = observations()
    o = lambda key: obs.get(key, "")  # noqa: E731
    out = [
        "# Experiment Results 6",
        "Plan 5: seeds 83–149 for the Ch 8 ablation arms, and ten-seed effort, event and run-time numbers for Ch 6. "
        "Generated by `posthoc_report6.py` from `artifacts/results/posthoc6/`, the plan-4 rows and the seed 11–71 "
        "rows of plans 1–3. Rates are seed-pooled unless marked; E_FA in mm.",
        "## Summary", o("Summary") or "_(observations not written yet)_",
        "## 0. Protocol", protocol(data),
        "## 1. Run record", run_record(),
        "## 2. Tier 1: Ch 6 analyses with ten seeds (main arms, plan-4 rows)",
        "### 2.1 Control effort", effort_section(data), o("Effort"),
        "### 2.2 Transient event metrics", event_section(data, cache), o("Event"),
        "### 2.3 Run-time cost per control decision", decision_time_section(data), o("Decision time"),
        "## 3. Held-out results of the ablation arms",
        "### 3.1 Ranged held-out 480", heldout_section(data, "heldout_ranged", list(ABL), "Ranged held-out"),
        "### 3.2 Fixed-severity held-out (0.15 s / 0.3 mm)",
        heldout_section(data, "heldout_fixed", list(ABL), "Fixed held-out"), o("Held-out"),
        "## 4. Gain occupancy", occupancy_section(), o("Occupancy"),
        "## 5. Frozen gains (four delay-bearing cells)", frozen_section(data), o("Frozen"),
        "## 6. History: one frame vs ten frames", history_section(data, cache), o("History"),
        "## 7. Training range: point vs range", range_section(data, cache), o("Range"),
        "## 8. Budget: GS-D-W 2M vs 300k", budget_section(data, cache), o("Budget"),
        "## 9. Consequences for the thesis text", o("Thesis") or "_(not written yet)_",
        "## 10. Files",
        "- **Results:** `artifacts/results/posthoc6/t{1,2,3}_*` (new seeds only); seed 11–71 rows from "
        "`posthoc/`, `posthoc2/`, `posthoc3/`; main-arm new seeds from `posthoc4/`.\n"
        "- **Models:** `artifacts/models/posthoc/plan5/`.\n"
        "- **Bootstrap cache:** `artifacts/results/posthoc6/paired_cache.json`.\n"
        "- **Code:** `slurm_plan5.sh`, `posthoc_report6.py`.\n- **Log:** `artifacts/results/plan5.log`.",
    ]
    return "\n\n".join(part for part in out if part) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "POSTHOC_RESULTS_6.md"))
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
