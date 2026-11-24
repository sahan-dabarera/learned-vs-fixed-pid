"""Thesis figures for experiment stage 2.

Reads the same rows as report_stage2.py (PID-RN from its final directories)
and writes PDF + PNG pairs to data/results/stage2/figures/.

Colour encoding (validated with the dataviz palette checker on white, in this
slot order): the seven main arms keep one hue each, plus a marker and line
style (PID square/dashed, PID-RN diamond/dash-dot, scheduler circle/solid,
direct triangle/dotted; nominal hollow, disturbed filled), so the figures also
read in grayscale. History figures use the stage-1 variant colours.

    python -m pathfollow.analysis.figures_stage2
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402
import numpy as np  # noqa: E402

from pathfollow.analysis import figures_stage1 as f1  # noqa: E402  (rcParams, axis helpers)
from pathfollow.analysis.figures_stage1 import (  # noqa: E402
    AXIS, INK2, MUTED, VAR_A, VARIANT_STYLE, completion_axis, efa_axis, grouped_bars, line, noise_axis,
    panel_title, series, top_legend,
)
from pathfollow.analysis.report_stage1 import SEEDS, estimate, index, levels_of  # noqa: E402
from pathfollow.analysis.report_stage2 import (  # noqa: E402
    DELAY_CELLS, FIG, H10, LEARNED, MAIN, PIDRN, SHORT, load, pair_rows, seed_completion,
)
from pathfollow.metrics import read_csv  # noqa: E402
from pathfollow.analysis.report_stage2 import P2  # noqa: E402

STYLE = {  # slot order as validated: blue, orange, teal, amber, pink, green, purple
    "pid_global_robust":        dict(color="#2a78d6", marker="s", ls="--", fill=True),
    "ppo_scheduler_ranged_h1":  dict(color="#eb6834", marker="o", ls="-", fill=True),
    "ppo_direct_ranged_h1":     dict(color="#1baf7a", marker="^", ls=":", fill=True),
    "pid_global_nominal":       dict(color="#eda100", marker="s", ls="--", fill=False),
    "ppo_scheduler_nominal_h1": dict(color="#e87ba4", marker="o", ls="-", fill=False),
    "ppo_direct_nominal_h1":    dict(color="#008300", marker="^", ls=":", fill=False),
    PIDRN:                      dict(color="#7d4fc4", marker="D", ls="-.", fill=True),
}
ORDER = MAIN


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def legend_top(fig, ax, ncol=4, y=1.0):
    handles, labels = ax.get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=ncol, y=y)


# ---------------------------------------------------------------- F2

CELLS = [(m, c) for m in ("stationary", "transient") for c in ("nominal", "delay", "noise", "combined")]
CELL_SHORT = {"nominal": "nom.", "delay": "delay", "noise": "noise", "combined": "comb."}
CELL_LABELS = [f"{CELL_SHORT[c]}\n{'stat.' if m == 'stationary' else 'trans.'}" for m, c in CELLS]


def fig_f2_main() -> None:
    idx = index(load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged"))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    width = 0.11
    x = np.arange(len(CELLS))
    for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
        ax = axes[row]
        for i, arm in enumerate(ORDER):
            st = STYLE[arm]
            ys = [float(idx[(arm, None, m, c)][key]) * scale for m, c in CELLS]
            face = st["color"] if st["fill"] else "white"
            ax.scatter(x + (i - 3) * width, ys, marker=st["marker"], s=22, facecolor=face, edgecolor=st["color"],
                       lw=1.0, zorder=3, label=SHORT[arm], clip_on=False)
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color="#f0efea", lw=0.6, zorder=0)
        if row == 0:
            completion_axis(ax)
            panel_title(ax, "Ranged held-out 480: completion")
        else:
            efa_axis(ax, top=300)
            panel_title(ax, "Failure-adjusted error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CELL_LABELS)
    axes[1].tick_params(axis="x", length=0)
    axes[0].axvline(3.5, color=AXIS, lw=0.8)
    axes[1].axvline(3.5, color=AXIS, lw=0.8)
    legend_top(fig, axes[0], ncol=7)
    fig.tight_layout(h_pad=1.0)
    save(fig, "f2_heldout_main")


def fig_f2_history() -> None:
    ep = load(["f2_heldout_ranged"], "pidrn_f2_heldout_ranged", file="episodes.csv")
    labels = [f"{'stat.' if m == 'stationary' else 'trans.'} {c}" for m, c in CELLS]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), sharey=True)
    for ax, arm in zip(axes.flat, ["ppo_scheduler_nominal_h1", "ppo_direct_nominal_h1",
                                   "ppo_scheduler_ranged_h1", "ppo_direct_ranged_h1"]):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for i, (mode, condition) in enumerate(CELLS):
            err, _ = pair_rows(ep, ep, arm, H10[arm], mode, condition, True)
            mean, low, high = estimate(err)
            y = len(CELLS) - 1 - i
            ax.scatter([float(np.mean(err[s])) * 1000 for s in SEEDS], [y] * len(SEEDS), s=11, facecolor="white",
                       edgecolor=MUTED, lw=0.8, zorder=2, label="per seed" if i == 0 else None)
            ax.plot([low * 1000, high * 1000], [y, y], color=VAR_A, lw=1.6, solid_capstyle="round", zorder=3,
                    label="95 % CI" if i == 0 else None)
            ax.scatter([mean * 1000], [y], s=24, color=VAR_A, edgecolor="white", lw=0.8, zorder=4,
                       label="seed-mean" if i == 0 else None)
        ax.set_xscale("symlog", linthresh=5)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(range(len(CELLS)))
        ax.set_yticklabels(labels[::-1])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("ΔE_FA, H1 − H10 (mm, symlog); < 0: H1 lower")
        panel_title(ax, f"{SHORT[arm]} vs {SHORT[H10[arm]]}")
    for ax in axes.flat:
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, -10), max(hi, 10)
        ax.set_xlim(lo, hi)
        ticks = [t for t in (-1000, -100, -10, 0, 10, 100, 1000) if lo <= t <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    legend_top(fig, axes[0, 0], ncol=3)
    fig.tight_layout(h_pad=1.4, w_pad=1.4)
    save(fig, "f2_history_paired_difference")


# ---------------------------------------------------------------- F3

def fig_f3() -> None:
    rows = load(["f3_delay_existing", "f3_delay_new"], "pidrn_f3_delay")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    ticks = [0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]
    for col, mode in enumerate(("stationary", "transient")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            ax.axvspan(0, 0.3, color="#f3f2ee", zorder=0, lw=0)
            ax.axvline(0.15, color=AXIS, lw=0.8, zorder=1)
            for arm in ORDER:
                x, y = series(idx, arm, levels, mode, "delay", key, scale)
                line(ax, x, y, STYLE[arm], SHORT[arm], ms=3.8)
            ax.set_xlim(-0.015, 0.815)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, f"Constant delay, {mode}")
            else:
                efa_axis(ax, ylabel=col == 0, top=1000)
                ax.set_xlabel("Command delay (s)")
    legend_top(fig, axes[0, 0], ncol=7)
    fig.tight_layout(h_pad=1.2, w_pad=1.6, rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, "Shaded: ranged training range (0–0.3 s). Vertical line: end of the old training range (0.15 s).",
             ha="center", va="bottom", fontsize=7, color=MUTED)
    save(fig, "f3_delay_sweep")


# ---------------------------------------------------------------- F4

def fig_f4() -> None:
    rows = load(["f4_dynamic_existing", "f4_dynamic_new"], "pidrn_f4_dynamic")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for col, variant in enumerate(("delay", "combined")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            for arm in ORDER:
                x, y = series(idx, arm, levels, "stationary", variant, key, scale)
                line(ax, x, y, STYLE[arm], SHORT[arm], ms=4.2)
            ax.set_xlim(0.12, 0.48)
            ax.xaxis.set_major_locator(FixedLocator(levels))
            ax.set_xticklabels([f"0–{lv:g}" for lv in levels])
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, "Dynamic delay" if variant == "delay" else "Dynamic delay + dynamic noise")
            else:
                efa_axis(ax, ylabel=col == 0, top=1000)
                ax.set_xlabel("Delay range, redrawn every 1–3 s (s)")
    legend_top(fig, axes[0, 0], ncol=7)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    save(fig, "f4_dynamic_delay")


# ---------------------------------------------------------------- F5

def f5_rows():
    return load(["f5_noise_sweep"], "pidrn_f5_noise_sweep", old_dirs=("e1_noise_coarse", "e1_noise_fine"))


def fig_f5(mode: str) -> None:
    rows = f5_rows()
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for col, condition in enumerate(("noise", "combined")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            ax.axvspan(0.1, 10, color="#f3f2ee", zorder=0, lw=0)
            for arm in ORDER:
                x, y = series(idx, arm, levels, mode, condition, key, scale)
                line(ax, x * 1000, y, STYLE[arm], SHORT[arm], ms=3.8)
            noise_axis(ax, xlabel=row == 1)
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, "Noise only" if condition == "noise" else "Noise + 0.15 s delay")
            else:
                efa_axis(ax, ylabel=col == 0)
    axes[1, 0].text(0.32, 600, "ranged training\nrange ≤ 10 mm", fontsize=6.5, color=MUTED, va="top")
    legend_top(fig, axes[0, 0], ncol=7)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    save(fig, f"f5_noise_sweep_{mode}")


def fig_f5_mechanism() -> None:
    rows = f5_rows()
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, (key, scale, label) in zip(axes, (("steer_total_variation_per_s", 1, "Steering TV/T (1/s, log)"),
                                              ("wheel_saturation_fraction", 100, "Wheel saturation (%)"))):
        ax.axvspan(0.1, 10, color="#f3f2ee", zorder=0, lw=0)
        for arm in ORDER:
            x, y = series(idx, arm, levels, "stationary", "noise", key, scale)
            line(ax, x * 1000, y, STYLE[arm], SHORT[arm], ms=3.8)
        noise_axis(ax)
        ax.set_ylabel(label)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.5, 1000)
    axes[0].yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_yticklabels(["1", "10", "100", "1000"])
    completion_axis(axes[1], ylabel=False)
    axes[1].set_ylabel("Wheel saturation (%)")
    panel_title(axes[0], "Steering activity")
    panel_title(axes[1], "Actuator saturation")
    legend_top(fig, axes[0], ncol=7)
    fig.tight_layout(w_pad=2.0)
    save(fig, "f5_mechanism_stationary_noise")


# ---------------------------------------------------------------- F6

def fig_f6() -> None:
    ep = read_csv(P2 / "f6_frozen_eval" / "episodes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, arm in zip(axes, ("ppo_scheduler_nominal_h1", "ppo_scheduler_ranged_h1")):
        variants = [("Dynamic gains", VARIANT_STYLE[0]), ("Frozen F-clean", VARIANT_STYLE[1]),
                    ("Frozen F-delay", VARIANT_STYLE[2])]
        values = []
        for controller in (arm, f"{arm}_frozen_clean", f"{arm}_frozen_delay"):
            per = seed_completion(ep, controller, DELAY_CELLS)
            values.append([100 * per[s] for s in SEEDS] + [100 * float(np.mean([per[s] for s in SEEDS]))])
        grouped_bars(ax, [f"s{s}" for s in SEEDS] + ["mean"], variants, values)
        ax.axvline(len(SEEDS) - 0.5, color=AXIS, lw=0.6)
        completion_axis(ax, ylabel=ax is axes[0])
        ax.set_ylim(0, 105)
        panel_title(ax, f"{SHORT[arm]} (H = 1): delay-bearing cells")
        ax.set_xlabel("Training seed")
    legend_top(fig, axes[0], ncol=3)
    fig.tight_layout(w_pad=1.6)
    save(fig, "f6_frozen_gain_completion")


def main() -> None:
    assert f1 and LEARNED  # imported for rcParams and checks
    print(f"writing to {FIG}")
    fig_f2_main()
    fig_f2_history()
    fig_f3()
    fig_f4()
    for mode in ("stationary", "transient"):
        fig_f5(mode)
    fig_f5_mechanism()
    fig_f6()


if __name__ == "__main__":
    main()
