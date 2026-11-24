"""Thesis figures for experiment stage 3 (gain box and budget).

Same colour / marker scheme as stage 2 (figures_stage2.STYLE). The scheduler
versions chosen by the stage-3 decision rule take the GS-D / GS-N slots; the
stage-2 GS-D is drawn as a thin grey line (or grey marker) for reference.

    python -m pathfollow.analysis.figures_stage3
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

from pathfollow.analysis import figures_stage2 as f2  # noqa: E402  (rcParams + stage-2 styles)
from pathfollow.analysis.figures_stage1 import (  # noqa: E402
    AXIS, INK2, MUTED, VAR_A, VAR_B, completion_axis, efa_axis, line, noise_axis, panel_title, series, top_legend,
)
from pathfollow.analysis.report_stage1 import SEEDS, index, levels_of  # noqa: E402
from pathfollow.analysis import report_stage3 as r3  # noqa: E402
from pathfollow.analysis.report_stage3 import (  # noqa: E402
    CHOSEN_D, CHOSEN_N, DRD, DRN, FIG, GSD, GSDW, GSN, M2, M3, MAIN, PIDRN, P3, SHORT, history, validation,
)

GREY = "#898781"
STYLE = dict(f2.STYLE)
STYLE[CHOSEN_D] = f2.STYLE["ppo_scheduler_ranged_h1"]
STYLE[CHOSEN_N] = f2.STYLE["ppo_scheduler_nominal_h1"]
REF_STYLE = dict(color=GREY, marker="o", ls="-", fill=False)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def ref_line(ax, x, y, label=None):
    ax.plot(x, y, color=GREY, lw=0.9, ls="-", marker="o", ms=2.6, mfc="white", mec=GREY, mew=0.7, zorder=2,
            label=label, clip_on=False)


def legend_with_ref(fig, ax, ncol=4):
    handles, labels = ax.get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=ncol)


# ---------------------------------------------------------------- G0

def kp_samples(directory: str, arm: str, manifest: str) -> dict[str, np.ndarray]:
    data = np.load(P3 / directory / "kp_samples.npz")
    out: dict[str, list] = {}
    for key in data.files:
        a, m, _seed, condition = key.split("|")
        if a == arm and m == manifest:
            out.setdefault(condition.split("/")[1], []).append(data[key])
    return {c: np.concatenate(v) for c, v in out.items()}


BINS = np.logspace(np.log10(0.8), np.log10(320), 46)
CONDITION_ORDER = ("nominal", "delay", "noise", "combined")


def kp_marks(ax, floors):
    for x, label in floors:
        ax.axvline(x, color=INK2, lw=0.8, ls="-", zorder=1)
    ax.axvline(2.0, color=f2.STYLE[PIDRN]["color"], lw=1.0, ls="-.", zorder=1)
    ax.axvline(8.0, color=f2.STYLE["pid_global_robust"]["color"], lw=1.0, ls="--", zorder=1)


def kp_axis(ax, xlabel):
    ax.set_xscale("log")
    ax.set_xlim(0.8, 320)
    ticks = [1, 2, 5, 8, 26, 100, 300]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    if xlabel:
        ax.set_xlabel("Applied Kp (log)")


def fig_g0_occupancy() -> None:
    samples = kp_samples("g0_occupancy", GSD, "held_out_ranged")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITION_ORDER):
        values = samples[condition]
        weights = np.full(len(values), 100.0 / len(values))
        ax.hist(values, bins=BINS, weights=weights, color=f2.STYLE[GSD]["color"], edgecolor="white", lw=0.4,
                zorder=2)
        kp_marks(ax, [(5, "floor"), (300, "ceiling")])
        kp_axis(ax, ax in axes[1])
        panel_title(ax, f"{condition} (both modes)")
        if ax in axes[:, 0]:
            ax.set_ylabel("Policy steps (%)")
    handles = [plt.Line2D([], [], color=INK2, lw=0.8, label="box bounds (5, 300)"),
               plt.Line2D([], [], color=f2.STYLE[PIDRN]["color"], lw=1.0, ls="-.", label="PID-RN Kp = 2"),
               plt.Line2D([], [], color=f2.STYLE["pid_global_robust"]["color"], lw=1.0, ls="--", label="PID-R Kp = 8")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, "Stage-2 GS-D (box Kp 5–300), ranged held-out 480, all five seeds. Bars at Kp = 5 are steps on "
             "the box floor.", ha="center", va="bottom", fontsize=7, color=MUTED)
    save(fig, "g0_gain_occupancy")


def fig_g4_occupancy_wide() -> None:
    old = kp_samples("g0_occupancy", GSD, "held_out_ranged")
    new = kp_samples("g4_occupancy_wide", GSDW, "held_out_ranged")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITION_ORDER):
        for values, color, label in ((old[condition], GREY, "GS-D (box Kp 5–300)"),
                                     (new[condition], VAR_A, "GS-D-W (box Kp 1–300)")):
            weights = np.full(len(values), 100.0 / len(values))
            ax.hist(values, bins=BINS, weights=weights, histtype="step", color=color, lw=1.3, zorder=3, label=label)
        kp_marks(ax, [(1, ""), (5, ""), (300, "")])
        kp_axis(ax, ax in axes[1])
        panel_title(ax, f"{condition} (both modes)")
        if ax in axes[:, 0]:
            ax.set_ylabel("Policy steps (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=f2.STYLE[PIDRN]["color"], lw=1.0, ls="-."),
                plt.Line2D([], [], color=f2.STYLE["pid_global_robust"]["color"], lw=1.0, ls="--")]
    labels += ["PID-RN Kp = 2", "PID-R Kp = 8"]
    top_legend(fig, handles, labels, ncol=4)
    fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, "Ranged held-out 480, all five seeds, 300k checkpoints. Black lines: box bounds 1, 5 and 300.",
             ha="center", va="bottom", fontsize=7, color=MUTED)
    save(fig, "g4_gain_occupancy_wide")


def curve_panel(ax_c, ax_e, root, arm, color, mark=None):
    val = validation()
    stacked_c, stacked_e, steps = [], [], None
    for s in SEEDS:
        h = history(root, arm, s)
        x = np.asarray([r["timesteps"] for r in h]) / 1000
        c = np.asarray([r["completion_rate"] for r in h]) * 100
        e = np.asarray([r["failure_adjusted_error_m"] for r in h]) * 1000
        ax_c.plot(x, c, color=color, lw=0.6, alpha=0.45, zorder=2)
        ax_e.plot(x, e, color=color, lw=0.6, alpha=0.45, zorder=2)
        sel = int(val[arm][s]["selected_timesteps"]) / 1000
        k = int(np.argmin(np.abs(x - sel)))
        ax_c.scatter([x[k]], [c[k]], s=14, facecolor="white", edgecolor=color, lw=0.9, zorder=4)
        ax_e.scatter([x[k]], [e[k]], s=14, facecolor="white", edgecolor=color, lw=0.9, zorder=4)
        n = min(len(c), len(stacked_c[0])) if stacked_c else len(c)
        stacked_c.append(c)
        stacked_e.append(e)
        steps = x if steps is None or len(x) < len(steps) else steps
    n = min(len(v) for v in stacked_c)
    ax_c.plot(steps[:n], np.mean([v[:n] for v in stacked_c], axis=0), color=color, lw=1.8, zorder=3)
    ax_e.plot(steps[:n], np.mean([v[:n] for v in stacked_e], axis=0), color=color, lw=1.8, zorder=3)
    if mark:
        for ax in (ax_c, ax_e):
            ax.axvline(mark, color=AXIS, lw=0.8, zorder=1)


def _curve_axes(ax_c, ax_e, title, xmax):
    completion_axis(ax_c, ylabel=False)
    ax_e.set_yscale("log")
    ax_e.set_ylim(3, 1000)
    ax_e.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300, 1000]))
    ax_e.yaxis.set_minor_locator(NullLocator())
    ax_e.set_yticklabels(["3", "10", "30", "100", "300", "1000"])
    ax_e.set_xlabel("Training steps (k)")
    ax_c.set_xlim(0, xmax)
    panel_title(ax_c, title)


def fig_g0_curves() -> None:
    arms = [(GSD, "GS-D (300k)"), (GSN, "GS-N (300k)"), (DRD, "DR-D (2M)"), (DRN, "DR-N (2M)")]
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.6), sharex="col", sharey="row")
    for col, (arm, title) in enumerate(arms):
        curve_panel(axes[0, col], axes[1, col], M2, arm, f2.STYLE[arm]["color"])
        _curve_axes(axes[0, col], axes[1, col], title, 310 if "scheduler" in arm else 2050)
    axes[0, 0].set_ylabel("Val. completion (%)")
    axes[1, 0].set_ylabel("Val. E_FA (mm, log)")
    handles = [plt.Line2D([], [], color=INK2, lw=0.6, alpha=0.6, label="seed"),
               plt.Line2D([], [], color=INK2, lw=1.8, label="seed mean"),
               plt.Line2D([], [], color=INK2, lw=0, marker="o", mfc="white", ms=4, label="kept checkpoint")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "g0_learning_curves")


def fig_g2_curves() -> None:
    arms = [("ppo_scheduler_ranged_h1_2m", "GS-D 2M"), ("ppo_scheduler_ranged_h1_wide_2m", "GS-D-W 2M")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.8), sharex=True, sharey="row")
    for col, ((arm, title), color) in enumerate(zip(arms, (GREY, VAR_A))):
        curve_panel(axes[0, col], axes[1, col], M3, arm, color, mark=300)
        _curve_axes(axes[0, col], axes[1, col], title, 2050)
    axes[0, 0].set_ylabel("Val. completion (%)")
    axes[1, 0].set_ylabel("Val. E_FA (mm, log)")
    handles = [plt.Line2D([], [], color=INK2, lw=0.6, alpha=0.6, label="seed"),
               plt.Line2D([], [], color=INK2, lw=1.8, label="seed mean"),
               plt.Line2D([], [], color=INK2, lw=0, marker="o", mfc="white", ms=4, label="kept checkpoint"),
               plt.Line2D([], [], color=AXIS, lw=0.8, label="300k (stage-2 budget)")]
    top_legend(fig, handles, [h.get_label() for h in handles], ncol=4)
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    save(fig, "g2_learning_curves_2m")


# ---------------------------------------------------------------- G4

def fig_g4_main() -> None:
    idx = index(r3.load("heldout_ranged"))
    cells, labels = f2.CELLS, f2.CELL_LABELS
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.5), sharex=True)
    width = 0.1
    x = np.arange(len(cells))
    order = MAIN
    for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
        ax = axes[row]
        for i, arm in enumerate(order):
            st = STYLE[arm]
            ys = [float(idx[(arm, None, m, c)][key]) * scale for m, c in cells]
            face = st["color"] if st["fill"] else "white"
            ax.scatter(x + (i - 3.5) * width, ys, marker=st["marker"], s=22, facecolor=face, edgecolor=st["color"],
                       lw=1.0, zorder=3, label=SHORT[arm], clip_on=False)
        ys = [float(idx[(GSD, None, m, c)][key]) * scale for m, c in cells]
        ax.scatter(x + 3.5 * width, ys, marker="o", s=16, facecolor="white", edgecolor=GREY, lw=0.9, zorder=3,
                   label="GS-D (stage 2)", clip_on=False)
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color="#f0efea", lw=0.6, zorder=0)
        ax.axvline(3.5, color=AXIS, lw=0.8)
        if row == 0:
            completion_axis(ax)
            panel_title(ax, "Ranged held-out 480: completion")
        else:
            efa_axis(ax, top=300)
            panel_title(ax, "Failure-adjusted error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].tick_params(axis="x", length=0)
    legend_with_ref(fig, axes[0], ncol=4)
    fig.tight_layout(h_pad=1.0)
    save(fig, "g4_heldout_main")


def _sweep(ax, idx, levels, mode, condition, key, scale, xscale=1.0):
    x, y = series(idx, GSD, levels, mode, condition, key, scale)
    ref_line(ax, x * xscale, y, "GS-D (stage 2)")
    for arm in MAIN:
        x, y = series(idx, arm, levels, mode, condition, key, scale)
        line(ax, x * xscale, y, STYLE[arm], SHORT[arm], ms=3.6)


def fig_g4_delay() -> None:
    rows = r3.load("delay")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    ticks = [0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]
    for col, mode in enumerate(("stationary", "transient")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            ax.axvspan(0, 0.3, color="#f3f2ee", zorder=0, lw=0)
            _sweep(ax, idx, levels, mode, "delay", key, scale)
            ax.set_xlim(-0.015, 0.815)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, f"Constant delay, {mode}")
            else:
                efa_axis(ax, ylabel=col == 0)
                ax.set_xlabel("Command delay (s)")
    legend_with_ref(fig, axes[0, 0], ncol=4)
    fig.tight_layout(h_pad=1.2, w_pad=1.6, rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, "Shaded: ranged training range (0–0.3 s).", ha="center", va="bottom", fontsize=7, color=MUTED)
    save(fig, "g4_delay_sweep")


def fig_g4_dynamic() -> None:
    rows = r3.load("dynamic")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    for col, variant in enumerate(("delay", "combined")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            _sweep(ax, idx, levels, "stationary", variant, key, scale)
            ax.set_xlim(0.12, 0.48)
            ax.xaxis.set_major_locator(FixedLocator(levels))
            ax.set_xticklabels([f"0–{lv:g}" for lv in levels])
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, "Dynamic delay" if variant == "delay" else "Dynamic delay + dynamic noise")
            else:
                efa_axis(ax, ylabel=col == 0)
                ax.set_xlabel("Delay range, redrawn every 1–3 s (s)")
    legend_with_ref(fig, axes[0, 0], ncol=4)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    save(fig, "g4_dynamic_delay")


def fig_g4_noise(mode: str) -> None:
    rows = r3.load("noise")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True)
    for col, condition in enumerate(("noise", "combined")):
        for row, (key, scale) in enumerate((("completion_rate", 100), ("failure_adjusted_error_m", 1000))):
            ax = axes[row, col]
            ax.axvspan(0.1, 10, color="#f3f2ee", zorder=0, lw=0)
            _sweep(ax, idx, levels, mode, condition, key, scale, xscale=1000)
            noise_axis(ax, xlabel=row == 1)
            if row == 0:
                completion_axis(ax, ylabel=col == 0)
                panel_title(ax, "Noise only" if condition == "noise" else "Noise + 0.15 s delay")
            else:
                efa_axis(ax, ylabel=col == 0)
    legend_with_ref(fig, axes[0, 0], ncol=4)
    fig.tight_layout(h_pad=1.2, w_pad=1.6, rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, f"{mode.capitalize()} mode. Shaded: ranged training range (≤ 10 mm).", ha="center",
             va="bottom", fontsize=7, color=MUTED)
    save(fig, f"g4_noise_sweep_{mode}")


def fig_g4_mechanism() -> None:
    rows = r3.load("noise")
    idx, levels = index(rows), levels_of(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, (key, scale, label) in zip(axes, (("steer_total_variation_per_s", 1, "Steering TV/T (1/s, log)"),
                                              ("wheel_saturation_fraction", 100, "Wheel saturation (%)"))):
        ax.axvspan(0.1, 10, color="#f3f2ee", zorder=0, lw=0)
        _sweep(ax, idx, levels, "stationary", "noise", key, scale, xscale=1000)
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
    legend_with_ref(fig, axes[0], ncol=4)
    fig.tight_layout(w_pad=2.0)
    save(fig, "g4_mechanism_stationary_noise")


def main() -> None:
    assert VAR_B and DRN  # imported for completeness
    print(f"writing to {FIG}")
    fig_g0_occupancy()
    fig_g0_curves()
    fig_g2_curves()
    fig_g4_main()
    fig_g4_delay()
    fig_g4_dynamic()
    for mode in ("stationary", "transient"):
        fig_g4_noise(mode)
    fig_g4_mechanism()
    fig_g4_occupancy_wide()


if __name__ == "__main__":
    main()
