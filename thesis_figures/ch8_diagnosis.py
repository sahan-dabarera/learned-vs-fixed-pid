"""Ch 8 figures: noise mechanism + derivative-filter check, frozen gains, history forest plot.

Sources (thesis names as in the README; GS-D = 'GS-D-W 2M', GS-N = 'GS-N-W',
GS-D (300k) = 'GS-D-W', narrow-range GS-D = 'GS-D' in stage3 / stage-2 'GS-D'):
- noise_mechanism.pdf: steering TV/T and wheel saturation, ten seeds, read from
  data/results/reports/stage4.md section 7 ('All metrics, 10 seeds', noise, stationary; arm
  names there are the thesis names); PID-R derivative-filter rows copied from
  data/results/reports/stage1.md, E1-F.
- frozen_gains.pdf: wide range (GS-D), ten seeds, from data/results/reports/stage5.md section 2 (cluster
  job 3366201); narrow range (stage-2 GS-D), ten seeds, from data/results/reports/stage6.md section 5;
  completion in the four delay-bearing cells.
- history_forest.pdf: data/results/reports/stage6.md section 6 (ten seeds: DR-N, DR-D, GS-D-W 300k and
  narrow GS-D pairs), Delta E_FA seed-mean with 95 % CI; filled = CI excludes 0 and at least six
  of ten seeds share the sign (per-seed tables of section 6).
Run: python thesis_figures/ch8_diagnosis.py   (writes the PDFs to thesis_figures/output/).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "results" / "reports"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
SRC4 = REPORTS / "stage4.md"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

MAP = {n: n for n in ("PID-N", "PID-R", "PID-RN", "GS-N", "GS-D", "DR-N", "DR-D")}
STYLE = {
    "PID-N": ("#1f4e9c", ":"), "PID-R": ("#1f4e9c", "--"), "PID-RN": ("#1f4e9c", "-"),
    "GS-N": ("#d9731a", "--"), "GS-D": ("#d9731a", "-"),
    "DR-N": ("#2e8b57", "--"), "DR-D": ("#2e8b57", "-"),
}
MARK = {"PID-N": "o", "PID-R": "s", "PID-RN": "D", "GS-N": "v", "GS-D": "^",
        "DR-N": "<", "DR-D": ">"}


def read_mechanism():
    """{arm: [(level, compl, tv, sat), ...]}: noise / stationary rows, ten-seed block of section 7."""
    lines = SRC4.read_text(encoding="utf-8").splitlines()
    sec7 = next(k for k, l in enumerate(lines) if l.startswith("## 7."))
    start = next(k for k in range(sec7, len(lines)) if "All metrics, 10 seeds" in lines[k])
    end = next(k for k in range(start, len(lines)) if lines[k].startswith("</details>"))
    out = {}
    for l in lines[start:end]:
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) == 10 and c[1] == "noise" and c[2] == "stationary":
            out.setdefault(c[0], []).append(
                (float(c[3].split()[0]), float(c[4]), float(c[7]), float(c[8])))
    return out


# PID-R derivative-filter check, noise only, stationary, completion % (stage1.md E1-F)
TAUD_LEVELS = [0.3, 1, 3, 10, 30, 100]
TAUD = {"0.01 s": [100, 100, 100, 0, 0, 0],
        "0.02 s": [100, 100, 100, 33.3, 0, 0],
        "0.05 s": [100, 100, 100, 100, 21.7, 0]}


def noise_mechanism():
    data = read_mechanism()
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.6))
    ticks = [0.3, 1, 3, 10, 30, 100]
    for name, lab in MAP.items():
        lv, _, tv, sat = zip(*data[lab])
        c, ls = STYLE[name]
        kw = dict(color=c, ls=ls, marker=MARK[name], ms=2.5, lw=1.1, label=name)
        ax[0].plot(lv, tv, **kw)
        ax[1].plot(lv, sat, **kw)
    for (lab, comp), ls in zip(TAUD.items(), ("-", "--", ":")):
        ax[2].plot(TAUD_LEVELS, comp, color="#1f4e9c", ls=ls, marker="s", ms=3, lw=1.2,
                   label=f"$\\tau_d$ = {lab}")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Steering TV/T (1/s, log)")
    ax[1].set_ylabel("Wheel saturation (%)")
    ax[2].set_ylabel("PID-R completion (%)")
    for a, t in zip(ax, ("(a) Steering activity", "(b) Wheel saturation",
                         "(c) PID-R, derivative filter")):
        a.axvspan(0.1, 10, color="0.92", lw=0)
        a.set_xscale("log")
        a.set_xticks(ticks)
        a.set_xticklabels([f"{t:g}" for t in ticks])
        a.minorticks_off()
        a.set_xlim(0.25, 120)
        a.set_xlabel("Noise (mm, log)")
        a.set_title(t, fontsize=8, fontweight="bold", loc="left")
        a.grid(True, color="0.88", lw=0.5)
    ax[1].set_ylim(-3, 103)
    ax[2].set_ylim(-3, 103)
    ax[2].legend(frameon=False, fontsize=7, loc="lower left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.savefig(OUT / "noise_mechanism.pdf")
    plt.close(fig)


# Completion in the four delay-bearing cells (%): dynamic, frozen-clean, frozen-delay per seed
# (panel title, seeds, values); last entry of each list is the mean over that panel's seeds
FROZEN = [
    ("GS-D, wide range",
     ["11", "23", "37", "53", "71", "83", "97", "109", "127", "149", "mean"],
     [(100, 90.8, 92.1), (97.9, 97.9, 98.8), (94.6, 15.4, 47.5), (96.7, 90.4, 95.8),
      (89.6, 67.9, 65.8), (96.2, 22.9, 62.1), (98.8, 100, 100), (95.4, 95.4, 95.4),
      (99.2, 95.8, 98.8), (59.2, 27.1, 33.3), (92.8, 70.4, 79.0)]),
    ("Narrow-range GS-D (300k)",
     ["11", "23", "37", "53", "71", "83", "97", "109", "127", "149", "mean"],
     [(93.3, 92.1, 93.3), (88.8, 91.2, 89.2), (93.8, 93.3, 93.8),
      (69.2, 96.7, 96.2), (95.8, 73.8, 82.5), (79.2, 27.9, 61.7), (92.1, 19.2, 37.1),
      (90.4, 23.3, 52.5), (96.7, 96.7, 96.7), (92.5, 32.5, 79.6), (89.2, 64.7, 78.2)]),
]


def frozen_gains():
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.9), sharey=True)
    w = 0.27
    cols = ("#d9731a", "#bdbdbd", "#6e6e6e")
    labs = ("Scheduling (gains change)", "Frozen-clean", "Frozen-delay")
    for a, (title, seeds, vals) in zip(ax, FROZEN):
        x = np.arange(len(seeds))
        v = np.array(vals)
        for j in range(3):
            a.bar(x + (j - 1) * w, v[:, j], w, color=cols[j], label=labs[j])
        a.set_xticks(x, [s if s != "mean" else "Mean" for s in seeds], fontsize=7)
        a.set_title(title, fontsize=8.5, fontweight="bold", loc="left")
        a.set_ylim(0, 105)
        a.axvline(len(seeds) - 1.5, color="0.6", lw=0.6)
        a.grid(axis="y", color="0.9", lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("Completion (%)")
    ax[1].set_ylabel("Completion (%)")
    ax[1].set_xlabel("Seed", fontsize=7.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "frozen_gains.pdf")
    plt.close(fig)


# H1 - H10, Delta E_FA (mm): (seed-mean, lo, hi, clear); rows: nominal, stat d/n/c, trans d/n/c.
# clear = CI excludes 0 and at least six of ten seeds share the sign (Ch 5 reading rule).
ROWLAB = ["Nominal", "Stat. delay", "Stat. noise", "Stat. comb.",
          "Trans. delay", "Trans. noise", "Trans. comb."]
HIST = {
    "DR-N": [(-26.19, -106.16, 23.71, 0), (-20.47, -121.26, 56.38, 0), (-25.50, -105.57, 24.41, 0),
             (-19.67, -118.13, 57.89, 0), (-26.55, -122.47, 33.71, 0), (-25.87, -105.94, 24.02, 0),
             (-21.85, -123.03, 55.32, 0)],
    "DR-D": [(-28.56, -90.63, 2.55, 0), (-44.23, -116.85, -6.11, 1), (-28.54, -90.64, 2.60, 0),
             (-47.13, -126.02, -4.11, 1), (-50.63, -149.71, -1.23, 1), (-28.65, -90.80, 2.54, 0),
             (-57.60, -163.75, -3.08, 1)],
    "Wide range\nGS-D (300k)": [(-1.21, -5.43, 2.95, 0), (-25.36, -66.81, 9.16, 0),
                               (2.60, -5.26, 10.65, 0), (-24.03, -62.66, 10.39, 0),
                               (-21.22, -46.08, -1.54, 1), (0.80, -2.76, 4.83, 0),
                               (-23.03, -56.25, 5.25, 0)],
    "Narrow range\nGS-D (300k)": [(0.40, -0.15, 0.96, 0), (-14.24, -35.22, 5.12, 0),
                                 (2.77, -0.75, 6.76, 0), (-11.88, -28.74, 4.48, 0),
                                 (-7.60, -21.15, 6.09, 0), (1.74, -0.45, 4.23, 0),
                                 (-7.39, -27.93, 10.70, 0)],
}


def history_forest():
    fig, ax = plt.subplots(1, 4, figsize=(6.3, 2.5), sharey=True)
    y = np.arange(7)[::-1]
    for a, (title, rows) in zip(ax, HIST.items()):
        a.axvline(0, color="0.4", lw=0.8)
        for yy, (m, lo, hi, clear) in zip(y, rows):
            a.plot([lo, hi], [yy, yy], color="#7a3e9d", lw=1.3)
            a.plot(m, yy, "o", ms=4.5, color="#7a3e9d", mfc="#7a3e9d" if clear else "white",
                   mew=1.1)
        a.axhline(5.5, color="0.85", lw=0.6)
        a.axhline(2.5, color="0.85", lw=0.6)
        a.set_title(title, fontsize=8.5, fontweight="bold")
        a.set_xlabel("Δ$E_{FA}$ (mm)", fontsize=7.5)
        a.grid(axis="x", color="0.92", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_yticks(y, ROWLAB)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "history_forest.pdf")
    plt.close(fig)


if __name__ == "__main__":
    noise_mechanism()
    frozen_gains()
    history_forest()
