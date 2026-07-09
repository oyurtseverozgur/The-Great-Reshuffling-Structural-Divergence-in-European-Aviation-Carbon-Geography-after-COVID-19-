# -*- coding: utf-8 -*-
"""
Figure generation for:
The Great Reshuffling: Structural Divergence in European Aviation
Carbon Geography after COVID-19 (2010-2026)

Reproduces every figure from the EUROCONTROL state-level CO2 panel.
Series are the January-anchored group aggregates and the balanced
39-state convergence panel described in the manuscript.

Output: four PNG files at 300 dpi, styled to match Office/Excel charts.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------
# Office / Excel visual theme
# ----------------------------------------------------------------------
BLUE   = "#4472C4"   # Office accent 1  -> carbon-priced core
ORANGE = "#ED7D31"   # Office accent 2  -> non-priced periphery
GREY   = "#A5A5A5"   # Office accent 3
TXT    = "#404040"   # dark grey text
GRID   = "#D9D9D9"   # light gridlines
AXIS   = "#BFBFBF"   # light axis lines

mpl.rcParams.update({
    "font.family": "Carlito",          # Calibri-metric compatible
    "font.size": 11,
    "text.color": TXT,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.color": TXT,
    "ytick.color": TXT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

YEARS = np.arange(2010, 2027)


def excel_axes(ax, ygrid=True):
    """Apply Excel-like styling: horizontal gridlines only, light spines."""
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0)


# ----------------------------------------------------------------------
# Data (January observations; group aggregates and balanced panel)
# ----------------------------------------------------------------------
# Group emission index, 2019 = 100
core_idx = np.array([81.8, 86.0, 82.5, 79.5, 82.0, 84.0, 87.0, 91.5, 95.5,
                     100.0, 98.0, 43.5, 66.5, 84.0, 92.0, 97.0, 100.6])
peri_idx = np.array([46.6, 51.5, 55.0, 62.0, 73.5, 80.0, 88.0, 83.5, 95.0,
                     100.0, 104.5, 54.0, 83.0, 113.0, 127.0, 134.5, 143.4])

# Absolute levels (Mt CO2): index x 2019 level (core 13.88, periphery 1.18)
core_mt = core_idx / 100.0 * 13.88
peri_mt = peri_idx / 100.0 * 1.18

# Non-priced periphery share of total European aviation CO2 (%)
peri_share = np.array([4.6, 4.9, 5.4, 6.2, 7.1, 7.5, 8.0, 7.2, 8.0,
                       7.9, 8.3, 11.4, 9.6, 10.3, 10.6, 10.6, 10.8])

# sigma-convergence: SD of log CO2 across the balanced 39-state panel
sigma = np.array([1.915, 1.940, 1.947, 1.937, 1.913, 1.884, 1.870, 1.840,
                  1.798, 1.808, 1.810, 1.852, 1.765, 1.755, 1.737, 1.719, 1.704])

# CO2 per flight (tonnes) by group
core_int = np.array([20.0, 20.15, 20.0, 20.55, 21.0, 21.45, 21.85, 21.95,
                     22.2, 22.65, 22.45, 24.0, 22.35, 22.25, 23.55, 23.95, 24.6])
peri_int = np.array([17.5, 17.55, 17.8, 18.3, 18.85, 19.4, 20.2, 19.95,
                     20.05, 21.25, 22.5, 23.55, 22.9, 23.75, 24.8, 24.5, 24.9])

# State-level change 2019 -> 2026 (%), balanced 39-state panel
states = [
    ("Albania", 164, "P"), ("Moldova", 110, "P"), ("Armenia", 97, "P"),
    ("Hungary", 90, "C"), ("Kosovo", 88, "P"), ("Serbia", 64, "P"),
    ("Bosnia and Herz.", 56, "P"), ("Slovakia", 50, "C"), ("Poland", 47, "C"),
    ("North Macedonia", 41, "P"), ("Türkiye", 40, "P"), ("Malta", 33, "C"),
    ("Belgium", 29, "C"), ("Romania", 24, "C"), ("Czechia", 21, "C"),
    ("Croatia", 19, "C"), ("Portugal", 19, "C"), ("Cyprus", 18, "C"),
    ("Greece", 18, "C"), ("Spain", 15, "C"), ("Luxembourg", 12, "C"),
    ("Bulgaria", 10, "C"), ("Canary Islands", 8, "C"), ("Austria", 6, "C"),
    ("Lithuania", 6, "C"), ("Italy", 6, "C"), ("Ireland", 5, "C"),
    ("France", 0, "C"), ("Switzerland", -1, "C"), ("Denmark", -3, "C"),
    ("United Kingdom", -4, "C"), ("Finland", -7, "C"), ("Netherlands", -7, "C"),
    ("Germany", -14, "C"), ("Montenegro", -15, "P"), ("Norway", -16, "C"),
    ("Slovenia", -25, "C"), ("Iceland", -28, "C"), ("Sweden", -37, "C"),
]

# beta-convergence scatter (annualised growth %/yr vs initial log CO2, 2019)
# balanced 39-state panel; core (C) and periphery (P)
beta_pts = [
    (8.6, -2.3, "P"), (9.0, 9.1, "P"), (9.15, 5.0, "P"), (9.3, 13.9, "P"),
    (9.5, 10.6, "P"), (9.8, 9.7, "P"), (10.3, 7.1, "P"), (13.9, 4.8, "P"),
    (12.4, -1.9, "P"),
    (9.1, 5.8, "C"), (9.35, -4.1, "C"), (9.75, 2.5, "C"), (10.25, 0.8, "C"),
    (10.35, 4.1, "C"), (10.5, -4.7, "C"), (10.7, 1.4, "C"), (10.85, 2.4, "C"),
    (11.2, 9.2, "C"), (11.5, 2.7, "C"), (11.65, 3.0, "C"), (11.75, 1.6, "C"),
    (12.3, -1.0, "C"), (12.35, 0.7, "C"), (12.4, -0.5, "C"), (12.45, 5.4, "C"),
    (12.5, -2.5, "C"), (12.5, -6.8, "C"), (12.7, 1.2, "C"), (12.75, 2.4, "C"),
    (12.85, 3.6, "C"), (13.1, -0.1, "C"), (13.6, -1.1, "C"), (13.9, 0.9, "C"),
    (14.3, 2.1, "C"), (14.45, 0.05, "C"), (14.7, -2.1, "C"), (14.8, -0.7, "C"),
    (11.0, 3.1, "C"), (13.2, 1.5, "C"),
]

COVID = 2020


# ----------------------------------------------------------------------
# Figure 1: (a) absolute levels, (b) indexed, (c) periphery share
# ----------------------------------------------------------------------
def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))

    # (a) absolute Mt
    ax = axes[0]
    ax.plot(YEARS, core_mt, color=BLUE, lw=2, marker="o", ms=4.5,
            mfc=BLUE, mec="white", mew=0.6, label="Carbon-priced core")
    ax.plot(YEARS, peri_mt, color=ORANGE, lw=2, marker="s", ms=4.5,
            mfc=ORANGE, mec="white", mew=0.6, label="Non-priced periphery")
    ax.set_title("(a)  January CO$_2$ emissions (Mt)")
    ax.set_ylabel("Mt CO$_2$")
    ax.set_ylim(0, 16)
    ax.axvline(COVID, color=GREY, ls=(0, (3, 3)), lw=1)
    excel_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.legend(loc="center left", frameon=True, framealpha=1,
              edgecolor=GRID, fontsize=8.5, handlelength=1.6)

    # (b) indexed 2019 = 100
    ax = axes[1]
    ax.plot(YEARS, core_idx, color=BLUE, lw=2, marker="o", ms=4.5,
            mfc=BLUE, mec="white", mew=0.6, label="Carbon-priced core")
    ax.plot(YEARS, peri_idx, color=ORANGE, lw=2, marker="s", ms=4.5,
            mfc=ORANGE, mec="white", mew=0.6, label="Non-priced periphery")
    ax.axhline(100, color=GREY, ls=(0, (3, 3)), lw=1)
    ax.axvline(COVID, color=GREY, ls=(0, (3, 3)), lw=1)
    ax.set_title("(b)  Index, 2019 = 100")
    ax.set_ylabel("Index (2019 = 100)")
    ax.set_ylim(30, 155)
    excel_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.legend(loc="lower center", frameon=True, framealpha=1,
              edgecolor=GRID, fontsize=8.5, handlelength=1.6)

    # (c) periphery share
    ax = axes[2]
    bars = ax.bar(YEARS, peri_share, color=BLUE, width=0.66,
                  edgecolor="white", linewidth=0.4)
    ax.set_title("(c)  Non-priced periphery share (%)")
    ax.set_ylabel("Share of total CO$_2$ (%)")
    ax.set_ylim(0, 13)
    excel_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    for x, v in zip(YEARS, peri_share):
        if x in (2010, 2019, 2026):
            ax.text(x, v + 0.25, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=TXT)

    fig.tight_layout(w_pad=1.6)
    fig.savefig("fig1_levels_share.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: (a) sigma-convergence, (b) beta-convergence scatter
# ----------------------------------------------------------------------
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))

    # (a) sigma
    ax = axes[0]
    ax.plot(YEARS, sigma, color=BLUE, lw=2, marker="o", ms=4.5,
            mfc=BLUE, mec="white", mew=0.6)
    ax.axvline(COVID, color=GREY, ls=(0, (3, 3)), lw=1)
    ax.set_title("(a)  $\\sigma$-convergence: SD of log CO$_2$")
    ax.set_ylabel("Standard deviation of log CO$_2$")
    ax.set_ylim(1.68, 1.97)
    excel_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(4))

    # (b) beta scatter
    ax = axes[1]
    xc = [p[0] for p in beta_pts if p[2] == "C"]
    yc = [p[1] for p in beta_pts if p[2] == "C"]
    xp = [p[0] for p in beta_pts if p[2] == "P"]
    yp = [p[1] for p in beta_pts if p[2] == "P"]
    ax.scatter(xc, yc, s=34, color=BLUE, edgecolor="white", linewidth=0.5,
               label="Carbon-priced core", zorder=3)
    ax.scatter(xp, yp, s=40, color=ORANGE, marker="s", edgecolor="white",
               linewidth=0.5, label="Non-priced periphery", zorder=3)
    # OLS fit line consistent with beta = -0.0100 (-1.00 %/yr per log unit)
    allx = np.array([p[0] for p in beta_pts])
    ally = np.array([p[1] for p in beta_pts])
    b, a = np.polyfit(allx, ally, 1)
    xf = np.linspace(allx.min(), allx.max(), 50)
    ax.plot(xf, a + b * xf, color=GREY, lw=1.8, ls=(0, (5, 3)),
            label=f"OLS fit ($\\beta$ = {b/100:.4f})", zorder=2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("(b)  $\\beta$-convergence, 2019$-$2026")
    ax.set_xlabel("Log CO$_2$, January 2019")
    ax.set_ylabel("Annualised growth (% / yr)")
    excel_axes(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=1,
              edgecolor=GRID, fontsize=8.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig("fig2_convergence.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: state-level change 2019 -> 2026
# ----------------------------------------------------------------------
def figure3():
    order = sorted(states, key=lambda s: s[1])   # ascending, so +164 on top
    names = [s[0] for s in order]
    vals = [s[1] for s in order]
    cols = [ORANGE if s[2] == "P" else BLUE for s in order]
    ypos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.0, 8.6))
    ax.barh(ypos, vals, color=cols, height=0.72,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Change in January CO$_2$, 2019 $\\rightarrow$ 2026 (%)")
    ax.axvline(0, color=TXT, lw=0.9)
    ax.set_xlim(-55, 185)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    for y, v in zip(ypos, vals):
        off = 2.5 if v >= 0 else -2.5
        ha = "left" if v >= 0 else "right"
        ax.text(v + off, y, f"{v:+d}%", va="center", ha=ha,
                fontsize=7.6, color=TXT)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Carbon-priced core"),
                       Patch(color=ORANGE, label="Non-priced periphery")],
              loc="lower right", frameon=True, framealpha=1,
              edgecolor=GRID, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_state_change.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: CO2 per flight crossover
# ----------------------------------------------------------------------
def figure4():
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(YEARS, core_int, color=BLUE, lw=2, marker="o", ms=4.5,
            mfc=BLUE, mec="white", mew=0.6, label="Carbon-priced core")
    ax.plot(YEARS, peri_int, color=ORANGE, lw=2, marker="s", ms=4.5,
            mfc=ORANGE, mec="white", mew=0.6, label="Non-priced periphery")
    ax.axvline(COVID, color=GREY, ls=(0, (3, 3)), lw=1)
    ax.set_ylabel("Average CO$_2$ per flight (tonnes)")
    ax.set_ylim(17, 25.6)
    excel_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(loc="upper left", frameon=True, framealpha=1,
              edgecolor=GRID, fontsize=9.5, handlelength=1.6)
    fig.tight_layout()
    fig.savefig("fig4_intensity.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("Figures written: fig1_levels_share.png, fig2_convergence.png, "
          "fig3_state_change.png, fig4_intensity.png")
