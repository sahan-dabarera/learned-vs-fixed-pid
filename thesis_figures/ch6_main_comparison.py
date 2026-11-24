"""Ch 6 figures: heatmap of the main test set and forest plot of paired differences.

Ten seeds. Numbers read from data/results/reports/stage4.md section 4 (arm names there are
the thesis names): seed-pooled table, per-seed table and the ten-seed column of the paired
differences. A dot is filled when the difference is clear in the sense of Ch 5: the 95 % CI
excludes 0 and most seeds (at least 6 of 10) share the sign of the mean.
Run: python thesis_figures/ch6_main_comparison.py   (writes the PDFs to thesis_figures/output/).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "results" / "reports"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
SRC = REPORTS / "stage4.md"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

CELLS = ["Nominal", "Delay", "Noise", "Combined", "Delay", "Noise", "Combined"]
CTRL = ["PID-N", "PID-R", "PID-RN", "GS-N", "GS-D", "DR-N", "DR-D"]
FIXED = {"PID-N", "PID-R", "PID-RN"}
# results-file column index (after the arm/seed columns) for each plotted cell:
# stat. nominal, stat. delay/noise/combined, tran. delay/noise/combined (tran. nominal = stat.)
COLS = [0, 1, 2, 3, 5, 6, 7]
KEYS = [("stationary", "nominal"), ("stationary", "delay"), ("stationary", "noise"),
        ("stationary", "combined"), ("transient", "delay"), ("transient", "noise"),
        ("transient", "combined")]


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _ce(cell):
    a, b = cell.split("/")
    return float(a), float(b)


def read_section():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    a = next(k for k, l in enumerate(lines) if l.startswith("## 4."))
    b = next(k for k, l in enumerate(lines) if l.startswith("## 5."))
    return lines[a:b]


def read_pooled(sec):
    i = next(k for k, l in enumerate(sec) if l.startswith("**Ranged held-out, 10 seeds**"))
    while not sec[i].startswith("| arm"):
        i += 1
    rows = {}
    i += 2
    while sec[i].startswith("|"):
        c = _cells(sec[i])
        rows[c[0]] = [_ce(c[1 + j]) for j in COLS]
        i += 1
    C = np.array([[v[0] for v in rows[n]] for n in CTRL])
    E = np.array([[v[1] for v in rows[n]] for n in CTRL])
    return C, E


def read_per_seed(sec):
    """{arm: {seed: [(compl, efa) per plotted cell]}}"""
    i = next(k for k, l in enumerate(sec) if "Per seed, learned arms" in l)
    while not sec[i].startswith("| arm"):
        i += 1
    out = {}
    i += 2
    while sec[i].startswith("|"):
        c = _cells(sec[i])
        out.setdefault(c[0], {})[c[1].split()[0]] = [_ce(c[2 + j]) for j in COLS]
        i += 1
    return out


def read_pairs(sec):
    """{(metric, pair, mode, cond): (mean, lo, hi)} from the ten-seed column."""
    out = {}
    metric = None
    for l in sec:
        if l.startswith("**ΔCompletion"):
            metric = "C"
        elif l.startswith("**ΔE_FA"):
            metric = "E"
        elif metric and l.startswith("|") and not l.startswith("| pair") and "---" not in l:
            c = _cells(l)
            m, ci = c[4].split(" [")
            lo, hi = ci.rstrip("]").split(", ")
            out[(metric, c[0], c[1], c[2])] = (float(m), float(lo), float(hi))
    return out


def fmt_c(v):
    return "100" if v == 100 else f"{v:.1f}"


def fmt_e(v):
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def heat_panel(ax, data, norm, cmap, fmt, title, colour_values):
    ax.imshow(colour_values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            rgba = plt.get_cmap(cmap)(norm(colour_values[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt(data[i, j]), ha="center", va="center",
                    color="white" if lum < 0.5 else "black", fontsize=8)
    ax.set_xticks(range(7), CELLS)
    ax.set_yticks(range(7), CTRL)
    ax.tick_params(length=0)
    for y in (2.5, 4.5):  # family separators
        ax.axhline(y, color="white", lw=3)
    for x in (0.5, 3.5):  # nominal | stationary | transient
        ax.axvline(x, color="white", lw=3)
    for x, lab in ((2, "Stationary"), (5, "Transient")):
        ax.text(x, -0.62, lab, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=22)
    for s in ax.spines.values():
        s.set_visible(False)


def heatmap(sec):
    C, E = read_pooled(sec)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.2))
    heat_panel(axes[0], C, PowerNorm(0.5, vmin=0, vmax=100), "YlOrRd", fmt_c,
               "Completion rate (%)", 100 - C)
    heat_panel(axes[1], E, LogNorm(vmin=1, vmax=200), "YlOrRd", fmt_e,
               "Failure-adjusted error (mm)", E)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(OUT / "main_heatmap.pdf")


PAIRS = ["GS-D − PID-RN", "GS-D − DR-D", "DR-D − PID-RN"]


def seed_diffs(pair, k, metric, pooled, per_seed):
    """Per-seed differences for cell k (first arm minus second, matched by seed)."""
    a, b = pair.split(" − ")
    j = 0 if metric == "C" else 1
    da = per_seed[a]
    if b in FIXED:
        ref = pooled[j][CTRL.index(b), k]
        return [v[k][j] - ref for v in da.values()]
    return [da[s][k][j] - per_seed[b][s][k][j] for s in da]


def build_pairs(sec):
    C, E = read_pooled(sec)
    per_seed = read_per_seed(sec)
    table = read_pairs(sec)
    res = {}
    for pair in PAIRS:
        res[pair] = {}
        for metric in ("C", "E"):
            rows = []
            for k, (mode, cond) in enumerate(KEYS):
                m, lo, hi = table.get((metric, pair, mode, cond), (0.0, 0.0, 0.0))
                d = seed_diffs(pair, k, metric, (C, E), per_seed)
                share = sum(np.sign(x) == np.sign(m) for x in d) if m != 0 else 0
                clear = int((lo > 0 or hi < 0) and share >= 6)
                rows.append((m, lo, hi, clear))
            res[pair][metric] = rows
    return res


ROWLAB = ["Nominal", "Stat. delay", "Stat. noise", "Stat. combined",
          "Trans. delay", "Trans. noise", "Trans. combined"]
COL = {"C": "#2a6fb0", "E": "#c4511f"}


def forest(sec):
    pairs = build_pairs(sec)
    sys.stdout.reconfigure(encoding="utf-8")
    for pair, d in pairs.items():  # printed for checking against the text
        print(pair, *(f"{m}{'*' if c else ''}" for m, _, _, c in d["C"] + d["E"]))
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.6), sharey=True)
    y = np.arange(7)[::-1]
    for r, (key, xlab) in enumerate((("C", "Δ completion (pp)\n> 0 favours first"),
                                     ("E", "Δ$E_{FA}$ (mm)\n< 0 favours first"))):
        for c, (pair, d) in enumerate(pairs.items()):
            ax = axes[r, c]
            ax.axvline(0, color="0.4", lw=0.8)
            for yy, (m, lo, hi, clear) in zip(y, d[key]):
                ax.plot([lo, hi], [yy, yy], color=COL[key], lw=1.4)
                ax.plot(m, yy, "o", ms=5, color=COL[key],
                        mfc=COL[key] if clear else "white", mew=1.2)
            ax.axhline(5.5, color="0.85", lw=0.6)
            ax.axhline(2.5, color="0.85", lw=0.6)
            if r == 0:
                ax.set_title(pair, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlab, fontsize=7.5)
            ax.grid(axis="x", color="0.92", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[r, 0].set_yticks(y, ROWLAB)
    for ax in axes[0]:
        ax.set_xlim(-20, 5)
    for ax in axes[1]:
        ax.set_xlim(-13, 56)
    fig.tight_layout(w_pad=0.8, h_pad=1.2)
    fig.savefig(OUT / "main_paired_forest.pdf")


if __name__ == "__main__":
    section = read_section()
    heatmap(section)
    forest(section)
