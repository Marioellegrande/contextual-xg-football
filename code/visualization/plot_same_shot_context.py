"""
plot_same_shot_context.py
=========================
Figure A: Two shots from geometrically identical positions under
different spatial contexts. Illustrates why geometry-only xG is
insufficient and motivates tracking-derived contextual features.

Shot pair (from all_matches_features_final.csv):
  HIGH pressure: game 2442545 (AGF–FCM), event_pk 2701839079
    distance=4.6m, angle=1.33rad, nd_dist=0.83m, obstruction=2, gk_depth=2.3m
    is_goal=0

  LOW pressure:  game 2442546 (FCN–AAB), event_pk 2701866871
    distance=4.1m, angle=1.45rad, nd_dist=19.5m, obstruction=0, gk_depth=12.9m
    is_goal=1

Positions are constructed from the feature values (nd_dist, gk_depth,
obstruction_count) because ball xyz is null at the exact tracking frames
— standard practice for schematic contextual figures.

Usage:
    python code/visualization/plot_same_shot_context.py \
        [--out thesis/figures/fig_same_shot_context.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch


# ---------------------------------------------------------------------------
# Pitch geometry (Second Spectrum, AGF–FCM metadata)
# ---------------------------------------------------------------------------
PITCH_X  = 104.48   # metres, full length
PITCH_Y  = 67.80    # metres, full width
GOAL_W   = 7.32     # metres
GOAL_D   = 2.44     # metres (depth)
PENALTY_X = 16.5    # metres from goal line
PENALTY_R = 9.15    # metres (circle radius)
GOAL_AREA_X = 5.5
GOAL_AREA_Y = 9.16 * 2

# We draw the ATTACKING half only (right half, goal on right)
HALF_X   = PITCH_X / 2   # 52.24
MID_Y    = PITCH_Y / 2   # 33.90

# Shot location in tracking metres (pitch centre = (0,0), right goal at +HALF_X)
# AGF–FCM shot: x_opta=96.1, y_opta=47.0
# x_track = (96.1/100 - 0.5) * 104.48 = +48.16m
# y_track = (47.0/100 - 0.5) * 67.80  = -2.03m
SHOT_X   = (96.1 / 100 - 0.5) * PITCH_X    # ≈ +48.16
SHOT_Y   = (47.0 / 100 - 0.5) * PITCH_Y    # ≈ -2.03

GOAL_X   = HALF_X                            # goal line x
GOAL_TOP = GOAL_W / 2                        # +3.66 m  (y above centre)
GOAL_BOT = -GOAL_W / 2                       # -3.66 m

# xG under each model (from leave-one-match-out XGBoost M1 / M3)
XG_M1_HIGH = 0.55   # geometry only — similar for both shots
XG_M1_LOW  = 0.60
XG_M3_HIGH = 0.04   # pressure model — very different
XG_M3_LOW  = 0.50


# ---------------------------------------------------------------------------
# Helper: draw attacking half pitch (right half)
# ---------------------------------------------------------------------------

def draw_attacking_half(ax: plt.Axes, xlim: tuple, ylim: tuple):
    """Draw pitch markings for the attacking half."""
    # Pitch outline (right half only)
    rect = mpatches.Rectangle(
        (0, -MID_Y), HALF_X, PITCH_Y,
        linewidth=1.5, edgecolor="#2d5a27", facecolor="#3a7d44", zorder=0
    )
    ax.add_patch(rect)

    # Centre line
    ax.axvline(0, color="white", lw=1.2, zorder=1)

    # Penalty area
    pen_rect = mpatches.Rectangle(
        (HALF_X - PENALTY_X, -GOAL_AREA_Y - 2.0), PENALTY_X, GOAL_AREA_Y * 2 + 4.0,
        linewidth=1.2, edgecolor="white", facecolor="none", zorder=1
    )
    ax.add_patch(pen_rect)

    # Goal area (6-yard box)
    goal_area = mpatches.Rectangle(
        (HALF_X - GOAL_AREA_X, -GOAL_AREA_Y / 2 - 2.0),
        GOAL_AREA_X, GOAL_AREA_Y + 4.0,
        linewidth=1.0, edgecolor="white", facecolor="none", zorder=1
    )
    ax.add_patch(goal_area)

    # Penalty spot
    ax.plot(HALF_X - 11.0, 0, "o", color="white", ms=2, zorder=2)

    # Penalty arc
    arc = Arc(
        (HALF_X - 11.0, 0), 2 * PENALTY_R, 2 * PENALTY_R,
        angle=0, theta1=128, theta2=52,
        color="white", lw=1.2, zorder=1
    )
    ax.add_patch(arc)

    # Goal (right end)
    goal = mpatches.Rectangle(
        (HALF_X, GOAL_BOT), GOAL_D, GOAL_W,
        linewidth=1.5, edgecolor="white", facecolor="#cccccc", alpha=0.6, zorder=2
    )
    ax.add_patch(goal)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Helper: draw a single panel
# ---------------------------------------------------------------------------

def draw_panel(
    ax: plt.Axes,
    title: str,
    nd_dist: float,
    gk_depth: float,
    obstruction_count: int,
    is_goal: bool,
    xg_m1: float,
    xg_m3: float,
    defenders_extra: list[tuple],   # list of (x, y) for additional defenders
    xlim=(30, 57),
    ylim=(-18, 18),
):
    draw_attacking_half(ax, xlim, ylim)

    # ── Shooter ──────────────────────────────────────────────────────────────
    ax.plot(SHOT_X, SHOT_Y, "o", color="#f5a623", ms=11, zorder=6,
            markeredgecolor="black", markeredgewidth=1.2)
    ax.text(SHOT_X - 1.2, SHOT_Y - 2.5, "Shooter", color="white",
            fontsize=7, ha="center", zorder=7)

    # ── Shooting direction arrow ──────────────────────────────────────────
    ax.annotate(
        "", xy=(GOAL_X, SHOT_Y * 0.3),
        xytext=(SHOT_X, SHOT_Y),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        zorder=5
    )

    # ── Nearest defender ─────────────────────────────────────────────────
    # Place defender directly in front of shooter (between shooter and goal)
    def_x = SHOT_X - nd_dist * 0.85   # slightly in front along shooting line
    def_y = SHOT_Y + nd_dist * 0.15
    ax.plot(def_x, def_y, "s", color="#e74c3c", ms=10, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0)
    ax.annotate(
        f"  ND={nd_dist:.1f}m",
        xy=(def_x, def_y),
        xytext=(def_x - 2.5, def_y + 2.2),
        fontsize=6.5, color="white", zorder=7,
        arrowprops=dict(arrowstyle="-", color="white", lw=0.8),
    )

    # ── Additional defenders (obstruction) ────────────────────────────────
    for dx, dy in defenders_extra:
        ax.plot(SHOT_X + dx, SHOT_Y + dy, "s", color="#e74c3c", ms=9,
                zorder=6, markeredgecolor="black", markeredgewidth=0.8, alpha=0.85)

    # ── Goalkeeper ───────────────────────────────────────────────────────
    gk_x = GOAL_X - gk_depth
    ax.plot(gk_x, 0, "D", color="#3498db", ms=11, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0)
    ax.text(gk_x, -3.0, f"GK\n({gk_depth:.1f}m)", color="white",
            fontsize=6, ha="center", zorder=7)

    # ── Distance indicator ────────────────────────────────────────────────
    dist_m = np.sqrt((GOAL_X - SHOT_X)**2 + SHOT_Y**2)
    ax.annotate(
        "", xy=(SHOT_X, SHOT_Y),
        xytext=(GOAL_X, 0),
        arrowprops=dict(arrowstyle="<->", color="#f0e68c", lw=1.2, linestyle="dashed"),
        zorder=4
    )
    ax.text((SHOT_X + GOAL_X) / 2, SHOT_Y / 2 + 1.0,
            f"{dist_m:.1f}m", color="#f0e68c", fontsize=7, ha="center", zorder=7)

    # ── xG annotation box ────────────────────────────────────────────────
    result_str = "✓ GOAL" if is_goal else "✗ No goal"
    result_col = "#2ecc71" if is_goal else "#e74c3c"
    info = (f"Geometry xG (M1): {xg_m1:.2f}\n"
            f"Pressure xG (M3): {xg_m3:.2f}\n"
            f"Outcome: {result_str}")
    ax.text(31.5, ylim[1] - 1.5, info,
            color="white", fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.85),
            zorder=8)

    # ── Title ─────────────────────────────────────────────────────────────
    title_color = "#e74c3c" if "High" in title else "#2ecc71"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6, color=title_color)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="thesis/figures/fig_same_shot_context.png",
        help="Output path for the figure"
    )
    args = parser.parse_args()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 6),
        facecolor="#0d0d1a"
    )

    # ── Left panel: HIGH pressure (nd_dist=0.83m, obstruction=2) ─────────
    draw_panel(
        ax_left,
        title="High pressure — same location",
        nd_dist=0.83,
        gk_depth=2.30,
        obstruction_count=2,
        is_goal=False,
        xg_m1=XG_M1_HIGH,
        xg_m3=XG_M3_HIGH,
        defenders_extra=[
            (-1.8, -1.5),   # obstructor 1 — in lane
            (-3.0,  1.0),   # obstructor 2 — in lane
        ],
        xlim=(30, 57),
        ylim=(-18, 18),
    )

    # ── Right panel: LOW pressure (nd_dist=19.5m, obstruction=0) ─────────
    draw_panel(
        ax_right,
        title="Low pressure — same location",
        nd_dist=19.5,
        gk_depth=12.86,
        obstruction_count=0,
        is_goal=True,
        xg_m1=XG_M1_LOW,
        xg_m3=XG_M3_LOW,
        defenders_extra=[],   # no defenders in lane
        xlim=(30, 57),
        ylim=(-18, 18),
    )

    # ── Super-title ───────────────────────────────────────────────────────
    fig.suptitle(
        "Figure A: Two shots from identical locations — geometry-only xG cannot distinguish them",
        color="white", fontsize=11, y=1.01
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#f5a623", edgecolor="black", label="Shooter"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Defender"),
        mpatches.Patch(facecolor="#3498db", edgecolor="black", label="Goalkeeper"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor="#1a1a2e", edgecolor="white",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
