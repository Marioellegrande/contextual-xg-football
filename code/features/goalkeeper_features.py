from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from frame_utils import (
    FrameKey,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames,
    resolve_shooter_defenders,
)


def compute_gk_features_for_shot(
    shooter_id: Optional[str],
    frame: Optional[FramePlayers],
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> Tuple[float, float, float]:
    """Return (gk_ball_distance, gk_depth, gk_lateral_offset).

    Coordinate system: physical meters, x ∈ [0, pitch_length], y ∈ [0, pitch_width].
    GK proxy: defender closest to the target goal line along the x-axis.
    """
    nan3 = (float("nan"), float("nan"), float("nan"))
    shooter_xy, defenders_xy = resolve_shooter_defenders(shooter_id, frame)
    if shooter_xy is None or defenders_xy is None or len(defenders_xy) == 0:
        return nan3

    sx, sy = float(shooter_xy[0]), float(shooter_xy[1])
    # Tracking data uses centered coordinates: x ∈ [-pitch_length/2, +pitch_length/2],
    # goals at x = ±pitch_length/2.  If the shooter is in the positive half they attack
    # the right goal (+half_len); negative half → left goal (-half_len).
    half_len = pitch_length_m / 2.0
    target_goal_x = half_len if sx >= 0.0 else -half_len

    # GK proxy: defender with x closest to target goal line
    gk = min(defenders_xy, key=lambda d: abs(d[0] - target_goal_x))
    gx, gy = float(gk[0]), float(gk[1])
    gk_ball_distance = math.hypot(gx - sx, gy - sy)
    gk_depth = abs(target_goal_x - gx)
    gk_lateral_offset = abs(gy)  # distance from pitch centre-line (centered coords: centre at y=0)
    return float(gk_ball_distance), float(gk_depth), float(gk_lateral_offset)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute goalkeeper positioning features from aligned tracking frames.")
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

    keys: List[FrameKey] = []
    for p, fidx in zip(df[per_col].tolist(), df[frame_col].tolist()):
        if pd.isna(p) or pd.isna(fidx):
            continue
        keys.append((int(p), int(fidx)))
    frames = load_required_frames(args.tracking_jsonl, keys)

    out_vals: List[Tuple[float, float, float]] = []
    for _, row in df.iterrows():
        shooter_id = _coerce_opta_id(row.get("player_id"))
        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            out_vals.append((float("nan"), float("nan"), float("nan")))
            continue
        key = (int(row[per_col]), int(row[frame_col]))
        out_vals.append(
            compute_gk_features_for_shot(
                shooter_id=shooter_id,
                frame=frames.get(key),
                pitch_length_m=float(args.pitch_length_m),
                pitch_width_m=float(args.pitch_width_m),
            )
        )

    df["gk_ball_distance"] = [v[0] for v in out_vals]
    df["gk_depth"] = [v[1] for v in out_vals]
    df["gk_lateral_offset"] = [v[2] for v in out_vals]
    df["__goalkeeper_method"] = "freeze_frame_goalkeeper_proxy"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()
