"""
plot_same_shot_context_v2.py
============================
Redesigned Figure 1.1:
  - Two pitch panels (top) showing spatial context
  - Shooting corridor visualisation (key new element)
  - Three-model bar chart (bottom):
      online calculator | geometry-only M1 | contextual M9
  - Actual computed xG values from leave-one-group-out XGBoost (no class weighting)

Shot pair (Danish Superliga 2024/25):
  HIGH pressure (missed):  game 2442545, event_pk 2701839079
    distance=4.56m, angle=1.33rad, nd_dist=0.83m, gk_depth=2.3m, obstruction=2
  LOW pressure  (scored):  game 2442546, event_pk 2701866871
    distance=4.07m, angle=1.45rad, nd_dist=19.5m, gk_depth=12.9m, obstruction=0

Model xG values (XGBoost, no class weighting, trained on all other matches):
  M1 (geometry):    Shot A = 0.29,  Shot B = 0.29   (near-identical)
  M9 (contextual):  Shot A = 0.07,  Shot B = 0.85   (12x separation)
Online calculator estimate (azcalculator.com, manual inputs):
  Shot A: 4.5m, very central, foot, no big chance, 1-2 defenders -> ~0.27
  Shot B: 4.1m, very central, foot, big chance, no defenders      -> ~0.63

Usage:
    python code/visualization/plot_same_shot_context_v2.py \
        [--out thesis/figures/fig_same_shot_context.png]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Arc, FancyArrowPatch, Wedge
import numpy as np

# ── Pitch constants (Second Spectrum, AGF–FCM metadata) ───────────────────────
PITCH_X   = 104.48
PITCH_Y   = 67.80
GOAL_W    = 7.32
GOAL_D    = 2.44
HALF_X    = PITCH_X / 2
MID_Y     = PITCH_Y / 2
PENALTY_X = 16.5
GOAL_AREA_X = 5.5

# Coordinate system: pitch centre = (0, 0), right goal at x = +HALF_X
# Shot position (same for both): opta x=96.1, y=47.0
SHOT_X = (96.1 / 100 - 0.5) * PITCH_X   # ≈ +48.16 m
SHOT_Y = (47.0 / 100 - 0.5) * PITCH_Y   # ≈ −2.03 m

GOAL_X   = HALF_X
GOAL_TOP = +GOAL_W / 2    # +3.66 m
GOAL_BOT = -GOAL_W / 2    # −3.66 m

# ── xG values ─────────────────────────────────────────────────────────────────
XG = {
    "online_A": 0.27, "online_B": 0.63,
    "m1_A":     0.29, "m1_B":     0.29,
    "m9_A":     0.07, "m9_B":     0.85,
}

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = "#0d1117"        # figure background
PITCH_FC = "#2d6a4f"        # grass green
PITCH_EC = "#40916c"        # lighter green for lines
WHITE    = "#e8e8e8"
COL_A    = "#e63946"        # Shot A colour (red — pressured, missed)
COL_B    = "#52b788"        # Shot B colour (green — open, scored)
COL_SHOOTER  = "#f4a261"    # orange
COL_DEF      = "#e63946"    # red
COL_GK       = "#457b9d"    # blue
CORRIDOR_A   = "#e63946"    # red corridor (blocked)
CORRIDOR_B   = "#52b788"    # green corridor (clear)
TEXT_DARK    = "#1a1a2e"


def draw_half_pitch(ax: plt.Axes, xlim, ylim):
    """Draw the attacking half (right half). xlim/ylim set the window."""
    # Grass
    rect = mpatches.Rectangle(
        (0, -MID_Y), HALF_X, PITCH_Y,
        linewidth=0, facecolor=PITCH_FC, zorder=0
    )
    ax.add_patch(rect)
    # Pitch outline
    ax.plot([0, HALF_X, HALF_X, 0], [-MID_Y, -MID_Y, MID_Y, MID_Y],
            color=WHITE, lw=1.4, zorder=1, alpha=0.6)
    ax.axvline(0, color=WHITE, lw=1.2, zorder=1, alpha=0.5)

    # Penalty area
    pen_y = GOAL_AREA_X + 5.5  # 11.0 m half-width
    pen = mpatches.Rectangle(
        (GOAL_X - PENALTY_X, -pen_y), PENALTY_X, 2 * pen_y,
        linewidth=1.2, edgecolor=WHITE, facecolor="none", alpha=0.55, zorder=1
    )
    ax.add_patch(pen)

    # Goal area (6-yard box)
    ga_hw = GOAL_W / 2 + 5.5   # ~9.16 m half-width
    ga = mpatches.Rectangle(
        (GOAL_X - GOAL_AREA_X, -ga_hw), GOAL_AREA_X, 2 * ga_hw,
        linewidth=1.0, edgecolor=WHITE, facecolor="none", alpha=0.45, zorder=1
    )
    ax.add_patch(ga)

    # Penalty spot
    ax.plot(GOAL_X - 11.0, 0, "o", color=WHITE, ms=2.2, zorder=2, alpha=0.6)

    # Penalty arc
    arc = Arc(
        (GOAL_X - 11.0, 0), 2 * 9.15, 2 * 9.15,
        angle=0, theta1=128, theta2=52,
        color=WHITE, lw=1.1, alpha=0.5, zorder=1
    )
    ax.add_patch(arc)

    # Goal
    goal = mpatches.Rectangle(
        (GOAL_X, GOAL_BOT), GOAL_D, GOAL_W,
        linewidth=1.8, edgecolor=WHITE, facecolor="#9ecae1", alpha=0.35, zorder=2
    )
    ax.add_patch(goal)
    # Goal posts
    for yy in [GOAL_TOP, GOAL_BOT]:
        ax.plot([GOAL_X, GOAL_X + GOAL_D], [yy, yy], color=WHITE, lw=2.0,
                zorder=3, alpha=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def draw_shooting_corridor(ax, is_blocked: bool):
    """Shade the triangle from shot location to goal posts."""
    colour = CORRIDOR_A if is_blocked else CORRIDOR_B
    alpha  = 0.18 if is_blocked else 0.14
    xs = [SHOT_X, GOAL_X, GOAL_X, SHOT_X]
    ys = [SHOT_Y, GOAL_TOP, GOAL_BOT, SHOT_Y]
    ax.fill(xs, ys, color=colour, alpha=alpha, zorder=2)
    # Corridor outline
    ax.plot([SHOT_X, GOAL_X], [SHOT_Y, GOAL_TOP], color=colour,
            lw=0.8, alpha=0.5, linestyle="--", zorder=3)
    ax.plot([SHOT_X, GOAL_X], [SHOT_Y, GOAL_BOT], color=colour,
            lw=0.8, alpha=0.5, linestyle="--", zorder=3)


def place_marker(ax, x, y, shape, color, size, label=None, label_dx=0,
                 label_dy=2.2, zorder=6):
    ax.plot(x, y, shape, color=color, ms=size, zorder=zorder,
            markeredgecolor=BG, markeredgewidth=1.0)
    if label:
        txt = ax.text(x + label_dx, y + label_dy, label,
                      color=WHITE, fontsize=6.5, ha="center",
                      zorder=zorder + 1, fontweight="bold")
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])


def draw_distance_arrow(ax, x1, y1, x2, y2, label, color="#f4d03f"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->", color=color,
                                lw=1.3, linestyle="dashed"))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    txt = ax.text(mx, my + 0.9, label, color=color, fontsize=7,
                  ha="center", fontweight="bold", zorder=9)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])


def info_box(ax, lines: list[tuple[str, str]], x, y, xlim, ylim):
    """Draw a rounded info box. lines = [(label, value), ...]"""
    text = "\n".join(f"{k}: {v}" for k, v in lines)
    ax.text(x, y, text, color=WHITE, fontsize=6.8, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#12192e",
                      edgecolor="#445566", alpha=0.88, linewidth=0.8),
            zorder=10)


def draw_panel(ax, title: str, title_color: str,
               nd_dist: float, gk_depth: float, obstruction_count: int,
               is_goal: bool, defenders_extra: list,
               xlim=(30, 57), ylim=(-16, 16)):

    draw_half_pitch(ax, xlim, ylim)
    draw_shooting_corridor(ax, is_blocked=(not is_goal))

    # ── Shooter ───────────────────────────────────────────────────────────────
    place_marker(ax, SHOT_X, SHOT_Y, "D", COL_SHOOTER, 13, zorder=8)
    ax.text(SHOT_X - 0.5, SHOT_Y - 2.8, "Shooter", color=COL_SHOOTER,
            fontsize=7, ha="center", fontweight="bold", zorder=9)

    # Shooting direction arrow
    ax.annotate("", xy=(GOAL_X - 0.3, SHOT_Y * 0.25),
                xytext=(SHOT_X - 0.5, SHOT_Y),
                arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1.4),
                zorder=5)

    # ── Nearest defender ──────────────────────────────────────────────────────
    # Clamp to visible area; if far, show at edge with dotted line + label
    max_visible_dist = xlim[1] - xlim[0] - 5   # keep within view
    show_nd_dist = min(nd_dist, max_visible_dist * 0.6)
    def_x = SHOT_X - show_nd_dist * 0.80
    def_y = SHOT_Y + show_nd_dist * 0.10
    # Clamp within xlim
    def_x = max(xlim[0] + 1.5, def_x)
    place_marker(ax, def_x, def_y, "s", COL_DEF, 12, zorder=8)

    if nd_dist > 12:
        # Far defender: just label in info box, show arrow to edge with label
        ax.annotate("",
                    xy=(def_x + 0.5, def_y),
                    xytext=(SHOT_X - 0.8, SHOT_Y + 0.2),
                    arrowprops=dict(arrowstyle="<->", color="#ff9999",
                                    lw=1.1, linestyle="dashed"),
                    zorder=4)
        txt = ax.text((SHOT_X + def_x) / 2 - 1, SHOT_Y + show_nd_dist * 0.05 + 2.5,
                      f"{nd_dist:.1f} m\n(off frame)", color="#ff9999",
                      fontsize=6.5, ha="center", fontweight="bold", zorder=9)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])
    else:
        draw_distance_arrow(ax, SHOT_X, SHOT_Y + 0.3, def_x, def_y + 0.3,
                            f"{nd_dist:.2f} m", color="#ff9999")

    # ── Extra defenders (obstructors in shooting lane) ─────────────────────────
    for dx, dy in defenders_extra:
        ax.plot(SHOT_X + dx, SHOT_Y + dy, "s", color=COL_DEF,
                ms=10, zorder=7, markeredgecolor=BG,
                markeredgewidth=0.8, alpha=0.85)

    # ── Goalkeeper ────────────────────────────────────────────────────────────
    gk_x = max(xlim[0] + 1.5, GOAL_X - gk_depth)
    place_marker(ax, gk_x, 0.0, "^", COL_GK, 13, zorder=8)
    ax.text(gk_x, -3.2, f"GK\n{gk_depth:.1f} m off line",
            color=COL_GK, fontsize=6.2, ha="center", zorder=9)

    # ── Distance to goal ──────────────────────────────────────────────────────
    dist_m = np.sqrt((GOAL_X - SHOT_X) ** 2 + SHOT_Y ** 2)
    ax.annotate("", xy=(GOAL_X, 0), xytext=(SHOT_X, SHOT_Y),
                arrowprops=dict(arrowstyle="<->", color="#f4d03f",
                                lw=1.2, linestyle=(0, (4, 2))),
                zorder=4)
    ax.text((SHOT_X + GOAL_X) / 2 - 0.5, SHOT_Y / 2 + 3.2,
            f"{dist_m:.1f} m", color="#f4d03f", fontsize=7.5,
            ha="center", fontweight="bold", zorder=9)

    # ── Outcome badge ─────────────────────────────────────────────────────────
    badge = "✓  SCORED" if is_goal else "✗  MISSED"
    badge_col = COL_B if is_goal else COL_A
    ax.text(xlim[0] + 0.6, ylim[1] - 0.5, badge,
            color=badge_col, fontsize=11, fontweight="bold",
            va="top", ha="left", zorder=10,
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

    # ── Info box (bottom of panel) ────────────────────────────────────────────
    lane = f"{obstruction_count} defender(s) in lane" if obstruction_count else "✓ Clear lane"
    nd_label = f"{nd_dist:.2f} m  ← very close!" if nd_dist < 2 else f"{nd_dist:.1f} m  ← far away"
    info_box(ax, [
        ("Nearest defender", nd_label),
        ("GK from goal line", f"{gk_depth:.1f} m"),
        ("Shooting lane",     lane),
    ], xlim[0] + 0.5, ylim[0] + 5.8, xlim, ylim)

    # ── Panel title ───────────────────────────────────────────────────────────
    ax.set_title(title, color=title_color, fontsize=11,
                 fontweight="bold", pad=8)


def draw_bar_chart(ax):
    """Grouped bar chart: 3 model groups × 2 shots."""

    groups  = ["Online\nCalculator\n(manual input)", "Geometry Only\n(M1)",
               "Contextual Model\n(M9 — tracking data)"]
    vals_a  = [XG["online_A"], XG["m1_A"], XG["m9_A"]]
    vals_b  = [XG["online_B"], XG["m1_B"], XG["m9_B"]]

    x       = np.arange(len(groups))
    width   = 0.32
    gap     = 0.04

    bars_a = ax.bar(x - width / 2 - gap / 2, vals_a, width,
                    color=COL_A, label="Shot A — pressured, missed",
                    zorder=3, edgecolor=BG, linewidth=0.5, alpha=0.92)
    bars_b = ax.bar(x + width / 2 + gap / 2, vals_b, width,
                    color=COL_B, label="Shot B — open space, scored",
                    zorder=3, edgecolor=BG, linewidth=0.5, alpha=0.92)

    # Value labels
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.2f}", ha="center", va="bottom",
                color=WHITE, fontsize=8.5, fontweight="bold", zorder=5)

    # ── Ratio / insight annotations ───────────────────────────────────────────
    # For each group: show separation bracket
    def draw_sep(gi, va, vb, label, color="#ffffff"):
        cx_a = x[gi] - width / 2 - gap / 2
        cx_b = x[gi] + width / 2 + gap / 2
        y_top = max(va, vb) + 0.06
        # Horizontal bracket
        ax.plot([cx_a, cx_b], [y_top, y_top], color=color, lw=1.2, zorder=5)
        ax.plot([cx_a, cx_a], [y_top - 0.01, y_top], color=color, lw=1.2, zorder=5)
        ax.plot([cx_b, cx_b], [y_top - 0.01, y_top], color=color, lw=1.2, zorder=5)
        ax.text((cx_a + cx_b) / 2, y_top + 0.015, label,
                ha="center", va="bottom", color=color,
                fontsize=7.5, fontweight="bold", zorder=6)

    draw_sep(0, vals_a[0], vals_b[0], "2.3× gap",  "#aaaaaa")
    draw_sep(1, vals_a[1], vals_b[1], "≈ identical", "#ff9999")
    draw_sep(2, vals_a[2], vals_b[2], "12× gap ✓", "#52b788")

    # ── Key insight box ───────────────────────────────────────────────────────
    ax.text(2.48, 0.55,
            "Tracking data\nautomatically\ncaptures what\nthe calculator\nneeds manually",
            color=WHITE, fontsize=7.5, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a3a2a",
                      edgecolor=COL_B, linewidth=1.0, alpha=0.9),
            zorder=6)
    # Arrow from box to M9 group
    ax.annotate("", xy=(x[2] + width / 2 + gap / 2 + 0.02, 0.55),
                xytext=(2.47, 0.55),
                arrowprops=dict(arrowstyle="<-", color=COL_B, lw=1.2))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_facecolor(BG)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=WHITE, fontsize=9.5)
    ax.set_ylabel("Predicted xG  (goal probability)", color=WHITE, fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=WHITE, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#445566")
    ax.tick_params(axis="x", bottom=False)
    ax.yaxis.grid(True, color="#333355", lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Reference line at 0.5
    ax.axhline(0.5, color="#556677", lw=0.8, linestyle=":", zorder=1, alpha=0.7)
    ax.text(-0.52, 0.502, "0.5", color="#778899", fontsize=7.5, va="bottom")

    # Legend
    leg = ax.legend(loc="upper left", framealpha=0.0,
                    labelcolor=WHITE, fontsize=9, ncol=2,
                    bbox_to_anchor=(0.0, 1.0))

    # Sub-annotation: manual input callout
    ax.text(x[0], -0.115,
            "⚠ Defender count entered manually\n   by the user — no spatial precision",
            ha="center", va="top", color="#aaaaaa",
            fontsize=7.2, style="italic", zorder=6)
    ax.text(x[2], -0.115,
            "✓ All features derived automatically\n   from 25 Hz tracking data",
            ha="center", va="top", color=COL_B,
            fontsize=7.2, fontweight="bold", zorder=6)

    ax.set_title("What does each model predict for these two shots?",
                 color=WHITE, fontsize=10.5, fontweight="bold", pad=10)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="thesis/figures/fig_same_shot_context.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(14, 10.5), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.6, 1.0],
        hspace=0.42,
        wspace=0.06,
        top=0.91, bottom=0.12, left=0.04, right=0.97
    )

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bar   = fig.add_subplot(gs[1, :])

    # ── Pitch panels ──────────────────────────────────────────────────────────
    draw_panel(
        ax_left,
        title="Shot A — High Pressure  (missed)",
        title_color=COL_A,
        nd_dist=0.83,
        gk_depth=2.30,
        obstruction_count=2,
        is_goal=False,
        defenders_extra=[(-1.9, -1.4), (-3.1, 1.1)],
        xlim=(30, 57), ylim=(-15, 15),
    )

    draw_panel(
        ax_right,
        title="Shot B — Open Space  (scored)",
        title_color=COL_B,
        nd_dist=19.5,
        gk_depth=12.86,
        obstruction_count=0,
        is_goal=True,
        defenders_extra=[],
        xlim=(30, 57), ylim=(-15, 15),
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    draw_bar_chart(ax_bar)

    # ── Shared geometry annotation between panels ──────────────────────────────
    fig.text(0.5, 0.975,
             "Two real shots from the Danish Superliga 2024/25 — "
             "identical geometry, opposite outcome",
             ha="center", va="top", color=WHITE,
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
             "Both shots taken from ≈ 4.5 m at a central angle. "
             "Spatial context — not shot location — determines the outcome.",
             ha="center", va="top", color="#aaaaaa", fontsize=9.5)

    # ── Legend (pitch markers) ─────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COL_SHOOTER, edgecolor=BG, label="Shooter"),
        mpatches.Patch(facecolor=COL_DEF,     edgecolor=BG, label="Defender"),
        mpatches.Patch(facecolor=COL_GK,      edgecolor=BG, label="Goalkeeper"),
        mpatches.Patch(facecolor=CORRIDOR_A,  edgecolor=BG, alpha=0.5,
                       label="Shooting corridor (blocked)"),
        mpatches.Patch(facecolor=CORRIDOR_B,  edgecolor=BG, alpha=0.5,
                       label="Shooting corridor (clear)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               facecolor="#12192e", edgecolor="#445566",
               labelcolor=WHITE, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
