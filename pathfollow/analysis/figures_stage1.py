"""Figures for stage 1 (sweeps on the sealed controllers).

Reads data/results/stage1 (same loaders as report_stage1.py) and writes
PDF + PNG pairs to data/results/stage1/figures/.

Colour encoding (validated with the dataviz palette checker on white):
* six sealed arms: one fixed hue each, plus a marker and line style per arm
  (PID square/dashed, scheduler circle/solid, direct triangle/dotted; nominal
  hollow, disturbed filled) so the figures also read in grayscale;
* variant figures (E1-F, E3, E4, E5): dark gray = sealed/reference,
  blue = first variant, orange = second variant; the arm is named per panel.

    python -m pathfollow.analysis.figures_stage1
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

from pathfollow.analysis.report_stage1 import (  # noqa: E402
    PH, SEEDS, SHORT, episodes, estimate, finished, index, levels_of, paired, summary,
)

OUT = PH / "figures"

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REF, VAR_A, VAR_B = "#52514e", "#2a78d6", "#eb6834"

ARM_STYLE = {  # fixed slot order: key arms take the strongest hues
    "pid_global_robust":       dict(color="#2a78d6", marker="s", ls="--", fill=True),
    "ppo_scheduler_disturbed": dict(color="#eb6834", marker="o", ls="-", fill=True),
    "ppo_direct_disturbed":    dict(color="#1baf7a", marker="^", ls=":", fill=True),
    "pid_global_nominal":      dict(color="#eda100", marker="s", ls="--", fill=False),
    "ppo_scheduler_nominal":   dict(color="#e87ba4", marker="o", ls="-", fill=False),
    "ppo_direct_nominal":      dict(color="#008300", marker="^", ls=":", fill=False),
}
ARM_ORDER = ["pid_global_nominal", "pid_global_robust", "ppo_scheduler_nominal",
             "ppo_scheduler_disturbed", "ppo_direct_nominal", "ppo_direct_disturbed"]
VARIANT_STYLE = [dict(color=REF, marker="s", ls="--"), dict(color=VAR_A, marker="o", ls="-"),
                 dict(color=VAR_B, marker="^", ls="-")]

NOISE_TICKS = [0.3, 1, 3, 10, 30, 100]
EFA_TICKS = [1, 3, 10, 30, 100, 300, 1000]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- helpers

def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def line(ax, x, y, style, label, lw=1.4, ms=4.2):
    face = style["color"] if style.get("fill", True) else "white"
    ax.plot(x, y, color=style["color"], ls=style["ls"], lw=lw, marker=style["marker"], ms=ms,
            mfc=face, mec=style["color"], mew=1.0, label=label, clip_on=False, zorder=3)


def completion_axis(ax, ylabel=True):
    ax.set_ylim(-3, 103)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    if ylabel:
        ax.set_ylabel("Completion (%)")


def efa_axis(ax, ylabel=True, top=1000):
    ax.set_yscale("log")
    ax.set_ylim(0.8, top)
    ax.yaxis.set_major_locator(FixedLocator([t for t in EFA_TICKS if t <= top]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in EFA_TICKS if t <= top])
    if ylabel:
        ax.set_ylabel("E_FA (mm, log)")


def noise_axis(ax, xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(0.25, 120)
    ax.xaxis.set_major_locator(FixedLocator(NOISE_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in NOISE_TICKS])
    if xlabel:
        ax.set_xlabel("Measurement noise σ (mm, log)")


def realistic_band(ax):
    ax.axvspan(10, 30, color="#f3f2ee", zorder=0, lw=0)


def series(idx, arm, levels, mode, condition, key, scale=1.0):
    xs, ys = [], []
    for lv in levels:
        row = idx.get((arm, lv, mode, condition))
        if row is not None and row.get(key, "") != "":
            xs.append(lv)
            ys.append(float(row[key]) * scale)
    return np.asarray(xs), np.asarray(ys)


def top_legend(fig, handles, labels, ncol, y=1.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               handlelength=2.6, columnspacing=1.4)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=6)


# ---------------------------------------------------------------- E1

def fig_e1(mode: str) -> None:
    rows = summary("e1_noise_coarse", "e1_noise_fine")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for col, condition in enumerate(("noise", "combined")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            realistic_band(ax)
            for arm in ARM_ORDER:
                x, y = series(idx, arm, levels, mode, condition, key, scale)
                line(ax, x * 1000, y, ARM_STYLE[arm], SHORT[arm])
            noise_axis(ax, xlabel=row == 1)
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, "Noise only" if condition == "noise" else "Noise + 0.15 s delay")
            else:
                efa_axis(ax, ylabel=col == 0)
    axes[1, 0].text(17.3, 1.25, "10–30 mm\nband", ha="center", va="center", fontsize=6.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=6)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    save(fig, f"e1_noise_sweep_{mode}")


def fig_e1_mechanism() -> None:
    rows = summary("e1_noise_coarse", "e1_noise_fine")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, (key, scale, label) in zip(axes, (("steer_total_variation_per_s", 1, "Steering TV/T (1/s, log)"),
                                              ("wheel_saturation_fraction", 100, "Wheel saturation (%)"))):
        realistic_band(ax)
        for arm in ARM_ORDER:
            x, y = series(idx, arm, levels, "stationary", "noise", key, scale)
            line(ax, x * 1000, y, ARM_STYLE[arm], SHORT[arm])
        noise_axis(ax)
        ax.set_ylabel(label)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.5, 1000)
    axes[0].yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].set_yticklabels(["1", "10", "100", "1000"])
    completion_axis(axes[1], ylabel=False)
    axes[1].set_ylabel("Wheel saturation (%)")
    panel_title(axes[0], "Steering activity")
    panel_title(axes[1], "Actuator saturation")
    handles, labels = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "e1_mechanism_stationary_noise")


def fig_e1f() -> None:
    rows = summary("e1_noise_coarse") + summary("e1f_taud")
    idx, levels = index(rows), levels_of(rows)
    arms = [("pid_global_robust", "τd = 0.01 s (sealed PID-R)"),
            ("pid_global_robust_taud0p02", "τd = 0.02 s"), ("pid_global_robust_taud0p05", "τd = 0.05 s")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, mode in zip(axes, ("stationary", "transient")):
        realistic_band(ax)
        for (arm, label), style in zip(arms, VARIANT_STYLE):
            x, y = series(idx, arm, levels, mode, "noise", "completion_rate", 100)
            line(ax, x * 1000, y, style, label)
        noise_axis(ax)
        completion_axis(ax, ylabel=mode == "stationary")
        panel_title(ax, f"PID-R, noise only, {mode}")
    handles, labels = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=3)
    fig.tight_layout(w_pad=1.6)
    save(fig, "e1f_pid_derivative_filter")


# ---------------------------------------------------------------- E2

def fig_e2() -> None:
    rows = summary("e2_delay")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for col, mode in enumerate(("stationary", "transient")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            ax.axvline(0.15, color=AXIS, lw=0.8, ls="-", zorder=1)
            for arm in ARM_ORDER:
                x, y = series(idx, arm, levels, mode, "delay", key, scale)
                line(ax, x, y, ARM_STYLE[arm], SHORT[arm])
            ax.set_xlim(-0.01, 0.41)
            ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.15, 0.2, 0.3, 0.4]))
            ax.set_xticklabels(["0", "0.1", "0.15", "0.2", "0.3", "0.4"])
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, f"Delay, {mode}")
            else:
                efa_axis(ax, ylabel=col == 0, top=1000)
                ax.set_xlabel("Command delay (s)")
    axes[0, 0].text(0.153, 4, "sealed\nstress level", fontsize=6.5, color=MUTED, va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=6)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    save(fig, "e2_delay_sweep")


# ---------------------------------------------------------------- held-out bar helpers

DELAY_CELLS = [("stationary", "delay"), ("stationary", "combined"), ("transient", "delay"), ("transient", "combined")]


def seed_completion(rows, controller, cells):
    """Completion (%) per seed over the given (mode, condition) cells."""
    by_seed = defaultdict(list)
    for row in rows:
        if row["controller"] == controller and (row["evaluation_mode"], row["condition"]) in cells:
            by_seed[row["training_seed"]].append(finished(row))
    return {s: 100 * float(np.mean(v)) for s, v in by_seed.items()}


def grouped_bars(ax, groups, variants, values, width=0.24):
    """values[variant][group] -> height. Bars carry a 2 pt white gap."""
    n = len(variants)
    x = np.arange(len(groups))
    for i, ((label, style), heights) in enumerate(zip(variants, values)):
        offs = (i - (n - 1) / 2) * width
        ax.bar(x + offs, heights, width=width, color=style["color"], edgecolor="white", linewidth=1.0,
               label=label, zorder=2)
        for xi, h in zip(x + offs, heights):
            if h < 0.5:
                ax.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=6, color=style["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- E3

def fig_e3() -> None:
    ho, fz = episodes("heldout_rerun"), episodes("e3_frozen_eval")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, arm in zip(axes, ("ppo_scheduler_nominal", "ppo_scheduler_disturbed")):
        variants = [("Dynamic (sealed)", VARIANT_STYLE[0]), ("Frozen F-clean", VARIANT_STYLE[1]),
                    ("Frozen F-delay", VARIANT_STYLE[2])]
        sources = [(ho, arm), (fz, f"{arm}_frozen_clean"), (fz, f"{arm}_frozen_delay")]
        values = []
        for rows, controller in sources:
            per = seed_completion(rows, controller, DELAY_CELLS)
            values.append([per[s] for s in SEEDS] + [float(np.mean([per[s] for s in SEEDS]))])
        grouped_bars(ax, [f"s{s}" for s in SEEDS] + ["mean"], variants, values)
        ax.axvline(len(SEEDS) - 0.5, color=AXIS, lw=0.6)
        completion_axis(ax, ylabel=arm == "ppo_scheduler_nominal")
        ax.set_ylim(0, 105)
        panel_title(ax, f"{SHORT[arm]}: delay-bearing held-out cells")
        ax.set_xlabel("Training seed")
    handles, labels = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=3)
    fig.tight_layout(w_pad=1.6)
    save(fig, "e3_frozen_gain_completion")


# ---------------------------------------------------------------- E4

E4_FAMILIES = [
    ("PID-R family", ["pid_global_robust", "pid_global_robust_noise_3mm", "pid_global_robust_noise_10mm"],
     ["PID-R (sealed)", "PID-RN, retuned at 3 mm", "PID-RN, retuned at 10 mm"]),
    ("GS-D family", ["ppo_scheduler_disturbed", "ppo_scheduler_noisetrained_3mm", "ppo_scheduler_noisetrained_10mm"],
     ["GS-D (sealed)", "GS-DN, trained to 3 mm", "GS-DN, trained to 10 mm"]),
    ("DR-D family", ["ppo_direct_disturbed", "ppo_direct_noisetrained_3mm", "ppo_direct_noisetrained_10mm"],
     ["DR-D (sealed)", "DR-DN, trained to 3 mm", "DR-DN, trained to 10 mm"]),
]
VARIANT_LEGEND = ["Sealed arm", "σ* = 3 mm variant", "σ* = 10 mm variant"]


def fig_e4_sweep(mode: str, condition: str) -> None:
    rows = summary("e1_noise_coarse", "e1_noise_fine", "e4_noise_sweep")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.4), sharex=True)
    for col, (title, arms, _) in enumerate(E4_FAMILIES):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            realistic_band(ax)
            for arm, style, label in zip(arms, VARIANT_STYLE, VARIANT_LEGEND):
                x, y = series(idx, arm, levels, mode, condition, key, scale)
                line(ax, x * 1000, y, style, label, ms=3.6)
            noise_axis(ax, xlabel=row == 1)
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, title)
            else:
                efa_axis(ax, ylabel=col == 0)
            if col:
                ax.tick_params(labelleft=False)
    for ax in axes[1]:
        ax.set_xlabel("σ (mm, log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.2, w_pad=0.8, rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.925), ncol=3, handlelength=2.6,
               columnspacing=1.4)
    fig.text(0.01, 0.985, f"{'Noise only' if condition == 'noise' else 'Noise + 0.15 s delay'}, {mode} mode",
             ha="left", va="top", fontsize=8.5, color=INK2)
    save(fig, f"e4_noise_sweep_{condition}_{mode}")


def fig_e4_heldout() -> None:
    ho = episodes("heldout_rerun")
    cells = [("stationary", "nominal"), *DELAY_CELLS]
    names = ["nom.\nstat.", "delay\nstat.", "comb.\nstat.", "delay\ntrans.", "comb.\ntrans."]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.5), sharey=True)
    for ax, (title, arms, _) in zip(axes, E4_FAMILIES):
        values = []
        for arm in arms:
            vals = []
            for cell in cells:
                per = seed_completion(ho, arm, [cell])
                vals.append(float(np.mean(list(per.values()))))
            values.append(vals)
        grouped_bars(ax, names, list(zip(VARIANT_LEGEND, VARIANT_STYLE)), values, width=0.26)
        completion_axis(ax, ylabel=ax is axes[0])
        ax.set_ylim(0, 105)
        panel_title(ax, title)
    handles, labels = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "e4_heldout_completion")


# ---------------------------------------------------------------- E5

def fig_e5() -> None:
    ho = episodes("heldout_rerun")
    cells = [(m, c) for m in ("stationary", "transient") for c in ("nominal", "delay", "noise", "combined")]
    labels = [f"{'stat.' if m == 'stationary' else 'trans.'} {c}" for m, c in cells]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, (h1, h10) in zip(axes, (("ppo_scheduler_disturbed_h1", "ppo_scheduler_disturbed"),
                                    ("ppo_direct_disturbed_h1", "ppo_direct_disturbed"))):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for i, (mode, condition) in enumerate(cells):
            err, _ = paired(ho, ho, h1, h10, mode, condition, match_seed=True)
            mean, low, high = estimate(err)
            seeds = [float(np.mean(err[s])) * 1000 for s in SEEDS]
            y = len(cells) - 1 - i
            ax.scatter(seeds, [y] * len(seeds), s=12, facecolor="white", edgecolor=MUTED, lw=0.8, zorder=2,
                       label="per seed" if i == 0 else None)
            ax.plot([low * 1000, high * 1000], [y, y], color=VAR_A, lw=1.6, solid_capstyle="round", zorder=3,
                    label="95 % CI" if i == 0 else None)
            ax.scatter([mean * 1000], [y], s=26, color=VAR_A, edgecolor="white", lw=0.8, zorder=4,
                       label="seed-mean" if i == 0 else None)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels(labels[::-1])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("ΔE_FA, H = 1 − H = 10 (mm)")
        panel_title(ax, f"{SHORT[h10]}: one frame vs ten-frame history")
    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.text(lo + 0.02 * (hi - lo), -0.95, "← H = 1 lower error", fontsize=6.5, color=MUTED, va="center")
        ax.set_ylim(-1.3, len(cells) - 0.5)
    handles, lbls = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, lbls, ncol=3)
    fig.tight_layout(w_pad=1.4)
    save(fig, "e5_history_paired_difference")


def main() -> None:
    print(f"writing to {OUT}")
    for mode in ("stationary", "transient"):
        fig_e1(mode)
    fig_e1_mechanism()
    fig_e1f()
    fig_e2()
    fig_e3()
    for condition in ("noise", "combined"):
        for mode in ("stationary", "transient"):
            fig_e4_sweep(mode, condition)
    fig_e4_heldout()
    fig_e5()


if __name__ == "__main__":
    main()
