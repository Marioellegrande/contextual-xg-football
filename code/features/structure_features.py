from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from frame_utils import (
    FrameKey,
    GOAL_WIDTH_M,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames,
    resolve_shooter_defenders,
)


HALF_GOAL_M = GOAL_WIDTH_M / 2.0  # half goal-width (m)
BODY_RADIUS_M = 0.5         # approximate half-width of a player body for blocking


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def _cross2d(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def _inside_sector(v_left: np.ndarray, v_right: np.ndarray, v_p: np.ndarray) -> bool:
    """True when direction v_p falls within the angular sector from v_left to v_right."""
    c1 = _cross2d(v_left, v_p)
    c2 = _cross2d(v_p, v_right)
    c_lr = _cross2d(v_left, v_right)
    if c_lr >= 0:
        return c1 >= 0 and c2 >= 0
    return c1 <= 0 and c2 <= 0


def compute_defender_dist_2(shooter_xy: np.ndarray, defenders_xy: np.ndarray) -> float:
    if defenders_xy is None or len(defenders_xy) < 2:
        return float("nan")
    dists = np.linalg.norm(defenders_xy - shooter_xy, axis=1)
    return float(np.partition(dists, 1)[1])


def compute_free_angle(
    shooter_xy: np.ndarray,
    defenders_xy: np.ndarray,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    body_radius: float = BODY_RADIUS_M,
) -> float:
    """Compute free (unobstructed) angle to goal in radians.

    Coordinate system: physical meters, x ∈ [0, pitch_length], y ∈ [0, pitch_width].
    Attacking direction is determined by the shooter's x position relative to the halfway line.
    A defender blocks a portion of the angle proportional to their angular width (body_radius / dist),
    but only when they are inside the goal sector AND between the shooter and the goal.
    """
    # Tracking data uses centered coordinates: x ∈ [-pitch_length/2, +pitch_length/2],
    # y ∈ [-pitch_width/2, +pitch_width/2], goal centre at y=0, posts at y = ±HALF_GOAL_M.
    half_len = pitch_length / 2.0

    # Determine which goal the shooter is attacking based on field position
    if shooter_xy[0] >= 0.0:
        # Right half → attacking goal at x = +half_len
        goal_left  = np.array([half_len, -HALF_GOAL_M], dtype=float)
        goal_right = np.array([half_len, +HALF_GOAL_M], dtype=float)
    else:
        # Left half → attacking goal at x = -half_len
        goal_left  = np.array([-half_len, +HALF_GOAL_M], dtype=float)
        goal_right = np.array([-half_len, -HALF_GOAL_M], dtype=float)

    v_left  = goal_left  - shooter_xy
    v_right = goal_right - shooter_xy
    full_angle = _angle(v_left, v_right)

    if full_angle < 1e-6 or defenders_xy is None or len(defenders_xy) == 0:
        return float(full_angle)

    # Mean distance from shooter to the two goalposts
    dist_to_goal = (np.linalg.norm(v_left) + np.linalg.norm(v_right)) / 2.0

    blocked = 0.0
    for d in defenders_xy:
        v_d = d - shooter_xy
        dist_d = float(np.linalg.norm(v_d))
        # Skip if defender is behind shooter or beyond the goal
        if dist_d < 1e-3 or dist_d >= dist_to_goal:
            continue
        # Skip if defender is outside the angular sector between goalposts
        if not _inside_sector(v_left, v_right, v_d):
            continue
        # Angular width this defender's body subtends from the shooter
        blocked += 2.0 * float(np.arctan2(body_radius, dist_d))

    return float(max(full_angle - blocked, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute structure features from aligned tracking freeze-frames.")
    ap.add_argument("--shots_aligned_csv", type=Path, required=True)
    ap.add_argument("--tracking_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--pitch_length_m", type=float, default=105.0)
    ap.add_argument("--pitch_width_m", type=float, default=68.0)
    args = ap.parse_args()

    df = pd.read_csv(args.shots_aligned_csv).copy()
    per_col, frame_col = _resolve_frame_columns(df)
    df[per_col] = pd.to_numeric(df[per_col], errors="coerce").astype("Int64")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").astype("Int64")
    keys = [
        (int(p), int(fidx))
        for p, fidx in zip(df[per_col].tolist(), df[frame_col].tolist())
        if not (pd.isna(p) or pd.isna(fidx))
    ]
    frames = load_required_frames(args.tracking_jsonl, keys)

    d2_vals: List[float] = []
    free_vals: List[float] = []
    for _, row in df.iterrows():
        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            d2_vals.append(float("nan"))
            free_vals.append(float("nan"))
            continue
        key = (int(row[per_col]), int(row[frame_col]))
        shooter_id = _coerce_opta_id(row.get("player_id"))
        shooter_xy, defenders_xy = resolve_shooter_defenders(shooter_id, frames.get(key))
        if shooter_xy is None or defenders_xy is None:
            d2_vals.append(float("nan"))
            free_vals.append(float("nan"))
            continue
        d2_vals.append(compute_defender_dist_2(shooter_xy, defenders_xy))
        free_vals.append(
            compute_free_angle(
                shooter_xy,
                defenders_xy,
                pitch_length=float(args.pitch_length_m),
                pitch_width=float(args.pitch_width_m),
            )
        )

    df["defender_dist_2"] = d2_vals
    df["free_angle"] = free_vals
    df["__structure_method"] = "second_defender_and_free_angle_from_tracking_frame"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()

