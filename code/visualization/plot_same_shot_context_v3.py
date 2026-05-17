"""
plot_same_shot_context_v3.py
============================
Clean academic-style Figure 1.1.

Layout:
  Top    : two pitch panels (top-down schematic, white background)
  Bottom : three horizontal-bar subplots (online calc | M1 | M9)

xG values (XGBoost, no class weighting, leave-one-group-out):
  M1 (geometry):   Shot A = 0.29,  Shot B = 0.29
  M9 (contextual): Shot A = 0.07,  Shot B = 0.85
Online calculator estimate (azcalculator.com):
  Shot A = 0.27,  Shot B = 0.63

Usage:
    python code/visualization/plot_same_shot_context_v3.py \
        [--out thesis/figures/fig_same_shot_context.png]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE       = "#ffffff"
PITCH_FC    = "#f7f7f7"     # very light grey pitch
PITCH_LINE  = "#444444"
GOAL_FC     = "#e0e0e0"
COL_A_DARK  = "#8b1a1a"     # shot A bars
COL_A_CORR  = "#f5c6c6"     # corridor fill A
COL_A_EDGE  = "#cc2200"     # corridor line / label A
COL_B_DARK  = "#1a5c1a"     # shot B bars
COL_B_CORR  = "#c8efc8"     # corridor fill B
COL_B_EDGE  = "#228833"     # corridor line / label B
COL_SHOOT   = "#e87722"     # shooter
COL_DEF     = "#cc2200"     # defender
COL_GK      = "#1a55aa"     # goalkeeper
GREY        = "#555555"
LGREY       = "#999999"

# ── Pitch schematic constants (metres, y=0 = goal line, y<0 = pitch) ─────────
GOAL_HALF  = 3.66
GOAL_DEPTH = 2.44          # "above" goal line in schematic
SHOT_Y     = -4.5          # shooter y-position
SHOT_X     = 0.0
PEN_HW     = 9.15          # penalty-area half-width (clipped to view)
PEN_D      = 16.5          # penalty-area depth
BOX_HW     = 5.5           # 6-yard box half-width (approx)
BOX_D      = 5.5

XLIM = (-12.0, 12.0)
YLIM = (-20.5, 3.5)


# ── Pitch drawing ─────────────────────────────────────────────────────────────

def draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # Pitch surface
    surf = mpatches.Rectangle(
        (XLIM[0], YLIM[0]), XLIM[1] - XLIM[0], abs(YLIM[0]),
        linewidth=1.6, edgecolor=PITCH_LINE, facecolor=PITCH_FC, zorder=0
    )
    ax.add_patch(surf)

    # Penalty area
    pen = mpatches.Rectangle(
        (-PEN_HW, -PEN_D), 2 * PEN_HW, PEN_D,
        linewidth=1.2, edgecolor=PITCH_LINE, facecolor="none", zorder=1
    )
    ax.add_patch(pen)

    # 6-yard box
    box = mpatches.Rectangle(
        (-BOX_HW, -BOX_D), 2 * BOX_HW, BOX_D,
        linewidth=1.1, edgecolor=PITCH_LINE, facecolor="none", zorder=1
    )
    ax.add_patch(box)

    # Penalty spot
    ax.plot(0, -11.0, "o", color=PITCH_LINE, ms=2.5, zorder=2)

    # Goal line
    ax.plot([-GOAL_HALF, GOAL_HALF], [0, 0],
            color=PITCH_LINE, lw=2.0, zorder=2)

    # Goal (rectangle above goal line)
    goal = mpatches.Rectangle(
        (-GOAL_HALF, 0), 2 * GOAL_HALF, GOAL_DEPTH,
        linewidth=2.0, edgecolor=PITCH_LINE, facecolor=GOAL_FC, zorder=2
    )
    ax.add_patch(goal)
    # crossbar line inside goal
    ax.plot([-GOAL_HALF, GOAL_HALF], [GOAL_DEPTH, GOAL_DEPTH],
            color=PITCH_LINE, lw=1.5, zorder=3)


def draw_corridor(ax: plt.Axes, is_blocked: bool) -> None:
    """Triangle from shooter to goal posts."""
    fc = COL_A_CORR if is_blocked else COL_B_CORR
    ec = COL_A_EDGE if is_blocked else COL_B_EDGE
    xs = [SHOT_X, -GOAL_HALF, GOAL_HALF]
    ys = [SHOT_Y,  0,          0]
    ax.fill(xs, ys, color=fc, alpha=0.55, zorder=2)
    ax.plot([SHOT_X, -GOAL_HALF], [SHOT_Y, 0], color=ec,
            lw=1.5, linestyle="--", zorder=3)
    ax.plot([SHOT_X,  GOAL_HALF], [SHOT_Y, 0], color=ec,
            lw=1.5, linestyle="--", zorder=3)


def place_defender(ax, x, y, size=9):
    ax.plot(x, y, "s", color=COL_DEF, ms=size, zorder=6,
            markeredgecolor="#550000", markeredgewidth=0.6)


def place_gk(ax, x, y, size=11):
    ax.plot(x, y, "^", color=COL_GK, ms=size, zorder=6,
            markeredgecolor="#0a2d6b", markeredgewidth=0.6)


def place_shooter(ax, x, y, size=13):
    ax.plot(x, y, "D", color=COL_SHOOT, ms=size, zorder=7,
            markeredgecolor="#7a3a00", markeredgewidth=0.8)


def info_box(ax, lines: list, x_ax: float, y_ax: float,
             ha="left", result_color=COL_A_DARK):
    """Draw a clean text box in axes coordinates."""
    text = "\n".join(lines)
    ax.text(
        x_ax, y_ax, text,
        transform=ax.transAxes,
        fontsize=8.2, va="top", ha=ha,
        color="#222222",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=WHITE,
                  edgecolor="#aaaaaa", linewidth=0.8, alpha=0.95),
        zorder=10
    )


# ── Shot A panel ──────────────────────────────────────────────────────────────

def draw_shot_A(ax: plt.Axes) -> None:
    draw_pitch(ax)
    draw_corridor(ax, is_blocked=True)

    # Obstructors in shooting lane
    place_defender(ax, -0.6, -2.8, size=9)
    place_defender(ax,  1.0, -3.5, size=9)

    # Nearest defender (0.83 m from shooter)
    nd_x, nd_y = 0.5, SHOT_Y + 0.83
    place_defender(ax, nd_x, nd_y, size=10)

    # Distance arrow: shooter → nearest defender
    ax.annotate(
        "", xy=(nd_x, nd_y - 0.05),
        xytext=(SHOT_X + 0.1, SHOT_Y + 0.05),
        arrowprops=dict(arrowstyle="<->", color=COL_DEF,
                        lw=1.2, shrinkA=5, shrinkB=5),
        zorder=5
    )
    ax.text(nd_x + 1.0, SHOT_Y + 0.45, "0.83 m",
            color=COL_DEF, fontsize=8, fontweight="bold", zorder=8)

    # Goalkeeper (2.3 m off line)
    place_gk(ax, 0.0, -2.3)

    # Shooter
    place_shooter(ax, SHOT_X, SHOT_Y)
    ax.text(SHOT_X, SHOT_Y - 1.4, "~4.5 m\n(central angle)",
            color=GREY, fontsize=7.5, ha="center", va="top", zorder=8)

    # Info box (left side)
    info_box(ax, [
        "Nearest defender: 0.83 m",
        "GK off line: 2.3 m",
        "Defenders in lane: 2",
        "Result: \u2717 Missed",
    ], x_ax=0.01, y_ax=0.70, ha="left", result_color=COL_A_DARK)

    # Titles (two text calls for two-line title)
    ax.text(0.5, 1.055, "Shot A – ", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5, fontweight="bold",
            color="#222222")
    ax.text(0.5, 1.055, "Missed", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, fontweight="bold",
            color=COL_A_DARK)
    ax.text(0.5, 1.020, "(high pressure)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, fontweight="bold",
            color="#222222")
    ax.text(0.5, 0.985, "xG context: pressured, blocked lane",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=LGREY, style="italic")


# ── Shot B panel ──────────────────────────────────────────────────────────────

def draw_shot_B(ax: plt.Axes) -> None:
    draw_pitch(ax)
    draw_corridor(ax, is_blocked=False)

    # Nearest defender is 19.5 m away — show at left edge with arrow + label
    nd_x_visible = XLIM[0] + 0.8
    nd_y_visible = SHOT_Y - 1.0
    place_defender(ax, nd_x_visible, nd_y_visible, size=9)
    ax.annotate(
        "", xy=(nd_x_visible + 0.3, nd_y_visible),
        xytext=(SHOT_X - 3.0, SHOT_Y),
        arrowprops=dict(arrowstyle="-|>", color=COL_DEF,
                        lw=1.0, linestyle="dashed"),
        zorder=4
    )
    ax.text(nd_x_visible + 1.8, nd_y_visible + 1.0, "19.5 m",
            color=COL_DEF, fontsize=8, fontweight="bold", zorder=8)

    # Goalkeeper (12.9 m off line — visible in view)
    place_gk(ax, 0.3, -12.9)

    # Shooter
    place_shooter(ax, SHOT_X, SHOT_Y)
    ax.text(SHOT_X, SHOT_Y - 1.4, "~4.5 m\n(central angle)",
            color=GREY, fontsize=7.5, ha="center", va="top", zorder=8)

    # Info box (right side)
    info_box(ax, [
        "Nearest defender: 19.5 m",
        "GK off line: 12.9 m",
        "Defenders in lane: 0",
        "Result: \u2713 Scored",
    ], x_ax=0.99, y_ax=0.70, ha="right", result_color=COL_B_DARK)

    # Titles
    ax.text(0.5, 1.055, "Shot B – ", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5, fontweight="bold",
            color="#222222")
    ax.text(0.5, 1.055, "Scored", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.5, fontweight="bold",
            color=COL_B_DARK)
    ax.text(0.5, 1.020, "(open space)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, fontweight="bold",
            color="#222222")
    ax.text(0.5, 0.985, "xG context: open, clear lane",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=LGREY, style="italic")


# ── Bar chart panels ──────────────────────────────────────────────────────────

def draw_bar_panel(ax: plt.Axes, title: str, val_a: float, val_b: float,
                   idx: int) -> None:
    """One horizontal bar chart panel."""
    ax.set_facecolor(WHITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")

    labels = ["Shot A – Missed", "Shot B – Scored"]
    values = [val_a, val_b]
    colors = [COL_A_DARK, COL_B_DARK]
    y_pos  = [1, 0]

    for yi, (v, c, lbl) in enumerate(zip(values, colors, labels)):
        bar = ax.barh(y_pos[yi], v, height=0.45, color=c,
                      edgecolor="none", zorder=3)
        # Value label inside or outside bar
        x_txt = v - 0.03 if v > 0.18 else v + 0.02
        ha_txt = "right" if v > 0.18 else "left"
        ax.text(x_txt, y_pos[yi], f"{v:.2f}",
                va="center", ha=ha_txt, fontsize=10.5,
                fontweight="bold",
                color=WHITE if v > 0.18 else c,
                zorder=5)

    # Gap bracket below bars
    gap = abs(val_b - val_a)
    ratio = val_b / val_a if val_a > 0 else float("inf")
    y_brack = -0.42
    ax.plot([val_a, val_b], [y_brack, y_brack],
            color=GREY, lw=1.2, zorder=4)
    ax.plot([val_a, val_a], [y_brack - 0.05, y_brack + 0.05],
            color=GREY, lw=1.2, zorder=4)
    ax.plot([val_b, val_b], [y_brack - 0.05, y_brack + 0.05],
            color=GREY, lw=1.2, zorder=4)
    ax.text((val_a + val_b) / 2, y_brack - 0.14,
            f"Gap: {gap:.2f} ({ratio:.1f}×)",
            ha="center", va="top", fontsize=8.5, color=GREY, zorder=5)

    # Axes
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.9, 1.7)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=8.5, colors=GREY)
    ax.set_title(f"{idx}) {title}", fontsize=9.5, fontweight="bold",
                 color="#222222", pad=6)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="thesis/figures/fig_same_shot_context.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(14, 10.5), facecolor=WHITE)

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.6, 1.0],
        hspace=0.52,
        top=0.94, bottom=0.10, left=0.04, right=0.97
    )

    # ── Top: two pitch panels ─────────────────────────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], wspace=0.22
    )
    ax_A = fig.add_subplot(gs_top[0])
    ax_B = fig.add_subplot(gs_top[1])
    draw_shot_A(ax_A)
    draw_shot_B(ax_B)

    # ── Bottom: three bar panels ──────────────────────────────────────────────
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1], wspace=0.38
    )
    ax_c1 = fig.add_subplot(gs_bot[0])
    ax_c2 = fig.add_subplot(gs_bot[1])
    ax_c3 = fig.add_subplot(gs_bot[2])

    draw_bar_panel(ax_c1,
                   "Online calculator\n(manual defender-count category)",
                   0.27, 0.63, idx=1)
    draw_bar_panel(ax_c2,
                   "Geometry-only model (M1)\n(shot location only)",
                   0.29, 0.29, idx=2)
    draw_bar_panel(ax_c3,
                   "Full contextual model (M9)\n(tracking-derived context)",
                   0.07, 0.85, idx=3)

    # Shared x-axis label
    fig.text(0.5, 0.065, "Predicted goal probability (xG)",
             ha="center", va="top", fontsize=10, color=GREY)

    # Section title for bar chart area
    fig.text(0.5, 0.425, "Predicted xG from different models",
             ha="center", va="bottom", fontsize=11,
             fontweight="bold", color="#222222")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        plt.Line2D([0], [0], marker="D", color="none",
                   markerfacecolor=COL_SHOOT, markersize=10,
                   markeredgecolor="#7a3a00", label="Shooter"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=COL_DEF, markersize=9,
                   markeredgecolor="#550000", label="Defender"),
        plt.Line2D([0], [0], marker="^", color="none",
                   markerfacecolor=COL_GK, markersize=10,
                   markeredgecolor="#0a2d6b", label="Goalkeeper"),
        plt.Line2D([0], [0], color=COL_A_EDGE, lw=1.5,
                   linestyle="--", label="Shooting corridor (blocked)"),
        plt.Line2D([0], [0], color=COL_B_EDGE, lw=1.5,
                   linestyle="--", label="Shooting corridor (clear)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=5,
        framealpha=0.0, labelcolor="#333333",
        fontsize=9, bbox_to_anchor=(0.5, 0.005)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
