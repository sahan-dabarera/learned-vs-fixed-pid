"""Trace figures (Ch 5, Ch 6, Ch 8): test paths, trajectory overlay, scheduler gains over time.

Source: data/results/stage5/traces_* and data/results/stage4/path_geometry (replays of the
evaluation checkpoints, described in
data/results/reports/stage5.md section 4). Thesis names: GS-D = ppo_scheduler_ranged_h1_wide_2m,
GS-N = ppo_scheduler_nominal_h1_wide, DR-D = ppo_direct_ranged_h1, DR-N = ppo_direct_nominal_h1,
PID-N / PID-R / PID-RN = pid_global_nominal / pid_global_robust / pid_global_robust_noise.
Representative seeds (clean-cell E_FA closest to the ten-seed median): GS-D 53, DR-D 37.

Writes:
- test_paths.pdf            all 20 test paths
- trajectory_overlay.pdf    path 10014, 0.5 m/s, stationary combined
- gsd_gains_over_time.pdf   GS-D seed 53: transient delay step, dynamic delay
Run: python thesis_figures/trace_figures.py   (writes the PDFs to thesis_figures/output/).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PATHS = RESULTS / "stage4" / "path_geometry"
TRACES = RESULTS / "stage5"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

CTRL = {"PID-N": "pid_global_nominal", "PID-R": "pid_global_robust",
        "PID-RN": "pid_global_robust_noise", "GS-N": "ppo_scheduler_nominal_h1_wide",
        "GS-D": "ppo_scheduler_ranged_h1_wide_2m", "DR-N": "ppo_direct_nominal_h1",
        "DR-D": "ppo_direct_ranged_h1"}
SEED = {"GS-D": 53, "DR-D": 37, "GS-N": 109, "DR-N": 11}
STYLE = {"PID-N": ("#1f4e9c", ":"), "PID-RN": ("#1f4e9c", "-"),
         "GS-D": ("#d9731a", "-"), "DR-D": ("#2e8b57", "-")}
TRACED = {"random_curvature_10019": "gentlest", "random_curvature_10001": "median",
          "random_curvature_10014": "hardest"}


def load_points():
    return pd.read_csv(PATHS / "path_points.csv")


def run(df, name, scen):
    sel = (df.controller == CTRL[name]) & (df.scenario_id == scen)
    if name in SEED:
        sel &= df.training_seed == SEED[name]
    return df[sel].sort_values("step")


# --------------------------------------------------------------------------- (a) test paths
def test_paths(pts):
    summ = pd.read_csv(PATHS / "path_summary.csv")
    test = summ[summ.split == "held_out"].reset_index(drop=True)
    fig, axs = plt.subplots(4, 5, figsize=(6.5, 5.6), sharex=True, sharey=True)
    for ax, (_, row) in zip(axs.flat, test.iterrows()):
        p = pts[(pts.split == "held_out") & (pts.path_id == row.path_id)]
        hi = row.path_id in TRACED
        ax.plot(p.x_m, p.y_m, color="#d9731a" if hi else "0.25", lw=1.4 if hi else 1.0)
        ax.plot(0, 0, "o", color="k", ms=2.5)
        pid = row.path_id.split("_")[-1]
        lab = f"{pid}" + (f" ({TRACED[row.path_id]})" if hi else "")
        ax.set_title(lab, fontsize=7, pad=2)
        ax.text(0.03, 0.04, f"max|κ| {row.max_abs_curvature:.2f}", transform=ax.transAxes,
                fontsize=6, color="0.35")
        ax.set_aspect("equal")
        ax.grid(lw=0.3, color="0.88")
    for ax in axs[-1]:
        ax.set_xlabel("x (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("y (m)")
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)
    fig.savefig(OUT / "test_paths.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- (b) trajectories
def trajectory_overlay(pts):
    h = pd.read_csv(TRACES / "traces_heldout" / "traces.csv.gz")
    scen = "held_out-random_curvature_10014-v0.5-stationary-combined-ranged"
    p = pts[(pts.split == "held_out") & (pts.path_id == "random_curvature_10014")]
    x, y = p.x_m.to_numpy(), p.y_m.to_numpy()
    # corridor edges: half-width d_corr = 1 m (Ch 2)
    dx, dy = np.gradient(x), np.gradient(y)
    n = np.hypot(dx, dy)
    nx, ny = -dy / n, dx / n

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.5, 3.0),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    ax0.fill(np.r_[x + nx, (x - nx)[::-1]],
             np.r_[y + ny, (y - ny)[::-1]], color="0.93", lw=0)
    ax0.plot(x, y, color="k", lw=2.4, alpha=0.25, label="Path")
    names = ["PID-N", "PID-RN", "GS-D", "DR-D"]
    for name in names:
        r = run(h, name, scen)
        c, ls = STYLE[name]
        lab = name + (f" (seed {SEED[name]})" if name in SEED else "")
        ax0.plot(r.x, r.y, color=c, ls=ls, lw=1.0, label=lab)
        ax1.plot(r.progress_m, 1000 * r.distance_m, color=c, ls=ls, lw=0.9, label=lab)
        if name == "PID-N":
            ax0.plot(r.x.iloc[-1], r.y.iloc[-1], "x", color=c, ms=6, mew=1.5)
    ax0.plot(0, 0, "o", color="k", ms=3)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("(a) Trajectories (grey band: corridor)", fontsize=8)
    ax1.set_xlabel("Progress along the path (m)")
    ax1.set_ylabel("Path distance (mm)")
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 2000)
    ax1.axhline(1000, color="0.5", lw=0.6, ls="--")
    ax1.text(7.95, 1150, "corridor edge", ha="right", fontsize=6.5, color="0.4")
    ax1.set_xlim(0, 8)
    ax1.set_title("(b) Distance to the path", fontsize=8)
    ax1.grid(lw=0.3, color="0.88")
    h0, l0 = ax0.get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.4)
    fig.savefig(OUT / "trajectory_overlay.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- (c) gains
def gains_over_time():
    h = pd.read_csv(TRACES / "traces_heldout" / "traces.csv.gz")
    d = pd.read_csv(TRACES / "traces_dynamic" / "traces.csv.gz")
    a = run(h, "GS-D", "held_out-random_curvature_10019-v0.5-transient-delay-ranged")
    b = run(d, "GS-D", "held_out-random_curvature_10001-v0.5-stationary-delay-dynamic-0p45s-delay")
    fig, axs = plt.subplots(3, 2, figsize=(6.5, 4.6), sharex="col",
                            gridspec_kw={"height_ratios": [0.8, 1.4, 1]})
    for j, (r, title) in enumerate([
            (a, "(a) Delay step at 6 s, path 10019"),
            (b, "(b) Changing delay (0–0.45 s), path 10001")]):
        t = r.t.to_numpy()
        axs[0, j].step(t, 1000 * r.actuator_delay_s, where="post", color="0.2", lw=1.0)
        axs[0, j].set_title(title, fontsize=8)
        axs[1, j].plot(t, r.kp, color="#d9731a", lw=0.5, alpha=0.35)
        kp_mean = r.kp.rolling(25, center=True, min_periods=1).mean()
        axs[1, j].plot(t, kp_mean, color="#d9731a", lw=1.3, label="$K_p$ (0.5 s mean)")
        axs[1, j].plot(t, r.kd, color="#7a3e9d", lw=0.9, label="$K_d$")
        axs[1, j].plot(t, r.ki, color="0.55", lw=0.8, label="$K_i$")
        axs[1, j].axhline(2.0, color="#1f4e9c", lw=0.7, ls="--", label="PID-RN $K_p$")
        axs[1, j].set_yscale("log")
        axs[1, j].set_ylim(0.02, 400)
        axs[2, j].plot(t, 1000 * r.distance_m, color="#d9731a", lw=0.9)
        axs[2, j].set_xlabel("Time (s)")
        axs[2, j].set_xlim(0, t.max())
        for ax in axs[:, j]:
            ax.grid(lw=0.3, color="0.88")
    axs[0, 0].set_ylabel("Delay (ms)")
    axs[1, 0].set_ylabel("Gain (log)")
    axs[2, 0].set_ylabel("Path dist. (mm)")
    hh, ll = axs[1, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout(rect=(0, 0.05, 1, 1), pad=0.4, h_pad=0.4)
    fig.savefig(OUT / "gsd_gains_over_time.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pts = load_points()
    test_paths(pts)
    trajectory_overlay(pts)
    gains_over_time()
