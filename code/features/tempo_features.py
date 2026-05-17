from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from frame_utils import (
    FrameKey,
    FrameState,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames_with_state as load_required_frames,
)


def _estimate_dt_seconds(curr: FrameState, prev: Optional[FrameState]) -> float:
    if prev is not None and curr.wallclock_ms is not None and prev.wallclock_ms is not None:
        dt = (curr.wallclock_ms - prev.wallclock_ms) / 1000.0
        if dt > 0:
            return dt
    # Default fallback for 25Hz feeds.
    return 0.04


def _speed(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]], dt: float) -> float:
    if a is None or b is None or dt <= 0:
        return float("nan")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) / dt)


def _player_xy(frame: FrameState, player_id: Optional[str]) -> Optional[Tuple[float, float]]:
    if player_id is None:
        return None
    xy = frame.home_by_opta.get(player_id)
    if xy is not None:
        return xy
    return frame.away_by_opta.get(player_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute short-window tempo features from tracking frames.")
    ap.add_argument("--shots_aligned_csv", type=Path, required=True)
    ap.add_argument("--tracking_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.shots_aligned_csv).copy()
    per_col, frame_col = _resolve_frame_columns(df)
    df[per_col] = pd.to_numeric(df[per_col], errors="coerce").astype("Int64")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").astype("Int64")

    keys: List[FrameKey] = []
    for p, fi in zip(df[per_col].tolist(), df[frame_col].tolist()):
        if pd.isna(p) or pd.isna(fi):
            continue
        per = int(p)
        frame = int(fi)
        keys.append((per, frame))
        keys.append((per, frame - 1))

    frames = load_required_frames(args.tracking_jsonl, keys)
    ball_speeds: List[float] = []
    shooter_speeds: List[float] = []
    for _, row in df.iterrows():
        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            ball_speeds.append(float("nan"))
            shooter_speeds.append(float("nan"))
            continue
        key = (int(row[per_col]), int(row[frame_col]))
        curr = frames.get(key)
        prev = frames.get((key[0], key[1] - 1))
        if curr is None:
            ball_speeds.append(float("nan"))
            shooter_speeds.append(float("nan"))
            continue
        dt = _estimate_dt_seconds(curr, prev)
        ball_speeds.append(_speed(curr.ball_xy, prev.ball_xy if prev is not None else None, dt))
        shooter_id = _coerce_opta_id(row.get("player_id"))
        shooter_curr = _player_xy(curr, shooter_id)
        shooter_prev = _player_xy(prev, shooter_id) if prev is not None else None
        shooter_speeds.append(_speed(shooter_curr, shooter_prev, dt))

    df["ball_speed_mps"] = ball_speeds
    df["shooter_speed_mps"] = shooter_speeds
    df["__tempo_method"] = "frame_t_minus_1_difference"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()
