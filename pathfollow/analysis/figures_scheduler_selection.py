"""Chapter-8 versions of two stage-3 figures, relabelled for the thesis text.

Thesis names: GS-D-W 2M -> "GS-D"; GS-D-W -> "GS-D (300k)"; stage-2 GS-D -> "narrow-range GS-D (300k)";
GS-D 2M -> "narrow-range GS-D (2M)"; "box" -> "gain range". No footers.

    python -m pathfollow.analysis.figures_scheduler_selection
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pathfollow import config  # noqa: E402
from pathfollow.analysis import figures_stage2 as f2  # noqa: E402
from pathfollow.analysis.figures_stage1 import INK2, VAR_A, top_legend, panel_title  # noqa: E402
from pathfollow.analysis.figures_stage3 import BINS, CONDITION_ORDER, GREY, _curve_axes, curve_panel, kp_axis, kp_samples  # noqa: E402
from pathfollow.analysis.report_stage3 import GSD, GSD2M, GSDW, GSDW2M, M2, M3, PIDRN  # noqa: E402

OUT = config.ROOT / "thesis_figures" / "output"
NARROW = "narrow-range GS-D (300k)"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf",):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_kp_histogram() -> None:
    old = kp_samples("g0_occupancy", GSD, "held_out_ranged")
    new = kp_samples("g4_occupancy_wide", GSDW, "held_out_ranged")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITION_ORDER):
        for values, color, label in ((old[condition], GREY, f"{NARROW}, gain range Kp 5–300"),
                                     (new[condition], VAR_A, "GS-D (300k), gain range Kp 1–300")):
            ax.hist(values, bins=BINS, weights=np.full(len(values), 100.0 / len(values)), histtype="step",
                    color=color, lw=1.3, zorder=3, label=label)
        for x in (1, 5, 300):
            ax.axvline(x, color=INK2, lw=0.8, zorder=1)
        ax.axvline(2.0, color=f2.STYLE[PIDRN]["color"], lw=1.0, ls="-.", zorder=1)
        ax.axvline(8.0, color=f2.STYLE["pid_global_robust"]["color"], lw=1.0, ls="--", zorder=1)
        kp_axis(ax, ax in axes[1])
        panel_title(ax, condition)
        if ax in axes[:, 0]:
            ax.set_ylabel("Policy steps (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK2, lw=0.8),
                plt.Line2D([], [], color=f2.STYLE[PIDRN]["color"], lw=1.0, ls="-."),
                plt.Line2D([], [], color=f2.STYLE["pid_global_robust"]["color"], lw=1.0, ls="--")]
    labels += ["gain-range limits (1, 5, 300)", "PID-RN Kp = 2", "PID-R Kp = 8"]
    top_legend(fig, handles, labels, ncol=3)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "ch8_kp_histogram_gain_range")


def fig_learning_curves() -> None:
    arms = [(M2, GSD, "narrow-range\nGS-D (300k)", GREY, 310), (M3, GSD2M, "narrow-range\nGS-D (2M)", GREY, 2050),
            (M3, GSDW, "\nGS-D (300k)", VAR_A, 310), (M3, GSDW2M, "\nGS-D", VAR_A, 2050)]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.7), sharey="row")
    for col, (root, arm, title, color, xmax) in enumerate(arms):
        curve_panel(axes[0, col], axes[1, col], root, arm, color, mark=300 if xmax > 1000 else None)
        _curve_axes(axes[0, col], axes[1, col], title, xmax)
        axes[0, col].tick_params(labelbottom=False)
        for ax in axes[:, col]:
            ax.set_xlim(0, xmax)
        if xmax > 1000:
            for ax in axes[:, col]:
                ax.set_xticks([0, 1000, 2000])
    axes[0, 0].set_ylabel("Val. completion (%)")
    axes[1, 0].set_ylabel("Val. E_FA (mm, log)")
    for ax in axes[0]:
        ax.title.set_fontsize(7.5)
    handles = [plt.Line2D([], [], color=INK2, lw=0.6, alpha=0.6), plt.Line2D([], [], color=INK2, lw=1.8),
               plt.Line2D([], [], color=INK2, lw=0, marker="o", mfc="white", ms=4),
               plt.Line2D([], [], color="#c3c2b7", lw=0.8)]
    top_legend(fig, handles, ["seed", "seed mean", "kept checkpoint", "300k steps"], ncol=4)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "ch8_learning_curves_candidates")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    fig_kp_histogram()
    fig_learning_curves()
