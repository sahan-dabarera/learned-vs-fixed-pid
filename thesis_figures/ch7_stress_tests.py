"""Ch 7 figures: constant-delay sweep, dynamic-delay test, noise sweep.

Ten seeds. Numbers are read directly from data/results/reports/stage4.md, sections 6.1, 6.2
and 7 (first seed-pooled table "completion % / E_FA mm" after each title = ten-seed table; the
five-seed copies follow later). Column labels there are already the thesis names.
Run: python thesis_figures/ch7_stress_tests.py   (writes the PDFs to thesis_figures/output/).
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "results" / "reports"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
SRC = REPORTS / "stage4.md"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

# thesis name -> results-file column label
MAP = {n: n for n in ("PID-N", "PID-R", "PID-RN", "GS-N", "GS-D", "DR-N", "DR-D")}
STYLE = {  # family colour, calibration / training by line style
    "PID-N": ("#1f4e9c", ":"), "PID-R": ("#1f4e9c", "--"), "PID-RN": ("#1f4e9c", "-"),
    "GS-N": ("#d9731a", "--"), "GS-D": ("#d9731a", "-"),
    "DR-N": ("#2e8b57", "--"), "DR-D": ("#2e8b57", "-"),
}
MARK = {"PID-N": "o", "PID-R": "s", "PID-RN": "D", "GS-N": "v", "GS-D": "^",
        "DR-N": "<", "DR-D": ">"}


def read_table(lines, start, title):
    """Return {column: [(level, compl, efa), ...]} for the first table after `title`."""
    i = start
    while lines[i].strip() != title:
        i += 1
    while not lines[i].startswith("|"):
        i += 1
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2
    out = {h: [] for h in header[1:]}
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        level = float(cells[0].split()[0])
        for h, c in zip(header[1:], cells[1:]):
            comp, efa = (float(x) for x in c.split("/"))
            out[h].append((level, comp, efa))
        i += 1
    return out


def section_start(lines, heading):
    return next(k for k, l in enumerate(lines) if l.startswith(heading))


def panel_pair(axc, axe, table, title, xscale, shade, xlabel, xticks=None, xticklabels=None):
    if shade is not None:
        for ax in (axc, axe):
            ax.axvspan(*shade, color="0.9", lw=0)
    for name, col in MAP.items():
        lv, cp, ef = zip(*table[col])
        c, ls = STYLE[name]
        kw = dict(color=c, ls=ls, marker=MARK[name], ms=3, lw=1.2, label=name)
        axc.plot(lv, cp, **kw)
        axe.plot(lv, ef, **kw)
    axc.set_title(title, fontsize=9, fontweight="bold", loc="left")
    axc.set_ylim(-4, 104)
    axc.set_yticks([0, 25, 50, 75, 100])
    axe.set_yscale("log")
    axe.set_ylim(1, 1000)
    for ax in (axc, axe):
        ax.set_xscale(xscale)
        ax.grid(True, color="0.85", lw=0.5)
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels if xticklabels else [f"{t:g}" for t in xticks])
            ax.minorticks_off()
        if xscale == "log":
            ax.set_xlim(0.25, 120)
    axe.set_xlabel(xlabel)


def four_panel(tab_a, tab_b, title_a, title_b, fname, **kw):
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex="col")
    panel_pair(ax[0, 0], ax[1, 0], tab_a, title_a, **kw)
    panel_pair(ax[0, 1], ax[1, 1], tab_b, title_b, **kw)
    ax[0, 0].set_ylabel("Completion (%)")
    ax[1, 0].set_ylabel("$E_{FA}$ (mm, log)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / fname)
    plt.close(fig)


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    s61 = section_start(lines, "### 6.1")
    s62 = section_start(lines, "### 6.2")
    s7 = section_start(lines, "## 7.")

    delay_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    four_panel(read_table(lines, s61, "**delay, stationary** (completion % / E_FA mm, seed-pooled)"),
               read_table(lines, s61, "**delay, transient** (completion % / E_FA mm, seed-pooled)"),
               "Stationary", "Transient (step at 6 s)", "delay_sweep.pdf",
               xscale="linear", shade=(0, 0.3), xlabel="Constant delay (s)", xticks=delay_ticks)

    dyn_ticks = [0.15, 0.3, 0.45]
    four_panel(read_table(lines, s62, "**delay, 10 seeds** (completion % / E_FA mm, seed-pooled)"),
               read_table(lines, s62, "**combined, 10 seeds** (completion % / E_FA mm, seed-pooled)"),
               "Changing delay", "Changing delay and noise", "dynamic_delay.pdf",
               xscale="linear", shade=None, xlabel="Delay range $[0, D_{max}]$ (s)",
               xticks=dyn_ticks, xticklabels=["0–0.15", "0–0.3", "0–0.45"])

    noise_ticks = [0.3, 1, 3, 10, 30, 100]
    for mode in ("stationary", "transient"):
        four_panel(read_table(lines, s7, f"**noise, {mode}** (completion % / E_FA mm, seed-pooled)"),
                   read_table(lines, s7, f"**combined, {mode}** (completion % / E_FA mm, seed-pooled)"),
                   "Noise only", "Noise + 0.15 s delay", f"noise_sweep_{mode}.pdf",
                   xscale="log", shade=(0.1, 10), xlabel="Noise standard deviation (mm, log)",
                   xticks=noise_ticks)


if __name__ == "__main__":
    main()
