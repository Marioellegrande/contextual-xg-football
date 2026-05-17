from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitchDims:
    """
    Pitch dimensions in meters.

    In this project we read pitch dimensions from Second Spectrum metadata
    (pitchLength, pitchWidth), and treat Opta event coordinates (x,y) as being
    on a [0,100] scale over the same pitch rectangle.
    """

    length_m: float
    width_m: float


GOAL_WIDTH_M = 7.32  # regulation goal width (meters)


def opta_xy_to_meters(x: float, y: float, pitch: PitchDims) -> Tuple[float, float]:
    """
    Convert Opta-style coordinates (x,y) in [0,100] to meters on the pitch rectangle.
    """
    return (x / 100.0) * pitch.length_m, (y / 100.0) * pitch.width_m


def compute_distance_m(x: float, y: float, pitch: PitchDims) -> float:
    """
    Euclidean distance from shot location to the centre of the goal (meters).

    Convention:
    - Attacking goal line is at x = 100 (i.e., x_m = pitch.length_m)
    - Goal centre is at y = 50 (i.e., y_m = pitch.width_m/2)
    """
    sx, sy = opta_xy_to_meters(x, y, pitch)
    gx, gy = pitch.length_m, pitch.width_m / 2.0
    return math.hypot(gx - sx, gy - sy)


def compute_angle_rad(x: float, y: float, pitch: PitchDims, goal_width_m: float = GOAL_WIDTH_M) -> float:
    """
    Shot angle as the angular opening to the goal (radians).

    Computed as the angle between the two vectors from the shot location to the two goalposts.

    Goalposts are placed at:
      (pitch.length_m, pitch.width_m/2 ± goal_width_m/2)
    """
    sx, sy = opta_xy_to_meters(x, y, pitch)

    post1 = (pitch.length_m, pitch.width_m / 2.0 - goal_width_m / 2.0)
    post2 = (pitch.length_m, pitch.width_m / 2.0 + goal_width_m / 2.0)

    ax, ay = post1[0] - sx, post1[1] - sy
    bx, by = post2[0] - sx, post2[1] - sy

    adotb = ax * bx + ay * by
    an = math.hypot(ax, ay)
    bn = math.hypot(bx, by)
    if an == 0.0 or bn == 0.0:
        return 0.0

    cosang = max(-1.0, min(1.0, adotb / (an * bn)))
    return math.acos(cosang)


def add_classic_geometry_features(
    shots_df: pd.DataFrame,
    pitch: PitchDims,
    x_col: str = "x",
    y_col: str = "y",
    out_distance_col: str = "distance_m",
    out_angle_col: str = "angle_rad",
) -> pd.DataFrame:
    """
    Add distance_m + angle_rad to a shot dataframe that contains Opta-style x,y in [0,100].

    Returns a copy of shots_df with two new columns.
    """
    if x_col not in shots_df.columns or y_col not in shots_df.columns:
        raise ValueError(f"shots_df must contain columns {x_col!r} and {y_col!r}")

    df = shots_df.copy()
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    # Vectorized distance to goal centre
    sx = (x / 100.0) * pitch.length_m
    sy = (y / 100.0) * pitch.width_m
    gx = pitch.length_m
    gy = pitch.width_m / 2.0
    df[out_distance_col] = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)

    # Vectorized angle: angular opening to goal between the two goalposts.
    # Both goalposts share the same x (goal line), so ax == bx = gx - sx.
    half_goal = GOAL_WIDTH_M / 2.0
    ax = gx - sx
    ay1 = (gy - half_goal) - sy
    ay2 = (gy + half_goal) - sy
    an = np.sqrt(ax**2 + ay1**2)
    bn = np.sqrt(ax**2 + ay2**2)
    valid = np.isfinite(sx) & np.isfinite(sy) & (an > 0) & (bn > 0)
    with np.errstate(invalid="ignore"):
        cosang = np.clip((ax * ax + ay1 * ay2) / (an * bn), -1.0, 1.0)
    df[out_angle_col] = np.where(valid, np.arccos(cosang), np.nan)

    return df

