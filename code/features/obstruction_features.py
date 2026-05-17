from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from frame_utils import (
    FrameKey,
    FramePlayers,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    GOAL_WIDTH_M,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames,
    resolve_shooter_defenders,
)


def _cross(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _inside_sector(v_left: Tuple[float, float], v_right: Tuple[float, float], v_p: Tuple[float, float]) -> bool:
    """Robust cone test using cross product sign consistency."""
    c1 = _cross(v_left, v_p)
    c2 = _cross(v_p, v_right)
    c_lr = _cross(v_left, v_right)
    if c_lr >= 0:
        return c1 >= 0 and c2 >= 0
    return c1 <= 0 and c2 <= 0


def compute_obstruction_for_shot(
    shooter_id: Optional[str],
    frame: Optional[FramePlayers],
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
    goal_width_m: float = GOAL_WIDTH_M,
) -> float:
    """Count defenders inside the angular sector between shooter and goalposts.

    Coordinate system: physical meters, x ∈ [0, pitch_length], y ∈ [0, pitch_width].
    """
    shooter_xy, defenders_xy = resolve_shooter_defenders(shooter_id, frame)
    if shooter_xy is None or defenders_xy is None or len(defenders_xy) == 0:
        return float("nan")

    sx, sy = float(shooter_xy[0]), float(shooter_xy[1])
    # Tracking data uses centered coordinates: x ∈ [-pitch_length/2, +pitch_length/2],
    # y ∈ [-pitch_width/2, +pitch_width/2], goal centre at y=0, posts at y = ±goal_width/2.
    half_len = pitch_length_m / 2.0
    target_goal_x = half_len if sx >= 0.0 else -half_len
    g1 = (target_goal_x, -goal_width_m / 2.0)
    g2 = (target_goal_x,  goal_width_m / 2.0)

    v1 = (g1[0] - sx, g1[1] - sy)
    v2 = (g2[0] - sx, g2[1] - sy)
    if math.hypot(*v1) == 0 or math.hypot(*v2) == 0:
        return float("nan")

    cnt = 0
    for d in defenders_xy:
        vp = (float(d[0]) - sx, float(d[1]) - sy)
        if _inside_sector(v1, v2, vp):
            cnt += 1
    return float(cnt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute obstruction_count from aligned tracking freeze-frames.")
    ap.add_argument("--shots_aligned_csv", type=Path, required=True)
    ap.add_argument("--tracking_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--pitch_length_m", type=float, default=105.0)
    ap.add_argument("--pitch_width_m", type=float, default=68.0)
    ap.add_argument("--goal_width_m", type=float, default=7.32)
    args = ap.parse_args()

    df = pd.read_csv(args.shots_aligned_csv).copy()
    per_col, frame_col = _resolve_frame_columns(df)
    df[per_col] = pd.to_numeric(df[per_col], errors="coerce").astype("Int64")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").astype("Int64")

    keys: List[FrameKey] = []
    for p, fidx in zip(df[per_col].tolist(), df[frame_col].tolist()):
        if pd.isna(p) or pd.isna(fidx):
            continue
        keys.append((int(p), int(fidx)))

    frames = load_required_frames(args.tracking_jsonl, keys)
    obstruction_vals: List[float] = []
    for _, row in df.iterrows():
        shooter_id = _coerce_opta_id(row.get("player_id"))
        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            obstruction_vals.append(float("nan"))
            continue
        key = (int(row[per_col]), int(row[frame_col]))
        obstruction_vals.append(
            compute_obstruction_for_shot(
                shooter_id=shooter_id,
                frame=frames.get(key),
                pitch_length_m=float(args.pitch_length_m),
                pitch_width_m=float(args.pitch_width_m),
                goal_width_m=float(args.goal_width_m),
            )
        )

    df["obstruction_count"] = obstruction_vals
    df["__obstruction_method"] = "freeze_frame_angular_sector_proxy"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()
