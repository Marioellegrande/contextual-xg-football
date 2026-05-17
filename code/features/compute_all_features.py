"""Unified feature computation — reads the tracking JSONL exactly once.

Replaces the chain of individual feature scripts (pressure, structure,
obstruction, goalkeeper, tempo, advanced_context) in the pipeline.
Instead of scanning the tracking file 6 separate times, this script:

  1. Reads the aligned shots CSV once.
  2. Collects all required frame keys (t and t-1 for tempo).
  3. Reads the tracking JSONL once via load_required_frames_with_state.
  4. Computes every feature group in a single pass over the shots table.
  5. Writes one combined output CSV.

Usage
-----
python code/features/compute_all_features.py \\
    --shots_aligned_csv  <path>/<match>_shots_aligned.csv \\
    --tracking_jsonl     <path>/<match>_SecondSpectrum_Data.jsonl \\
    --out_csv            <path>/<match>_features_final.csv
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from databallpy.features import get_pressure_on_player

from frame_utils import (
    FrameKey,
    FrameState,
    GOAL_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames_with_state,
    resolve_shooter_defenders,
)
from pressure_features import (
    _parse_radii_m,
    _radius_label_m,
    _to_databallpy_td_frame_and_column_id,
)
from goalkeeper_features import compute_gk_features_for_shot
from obstruction_features import compute_obstruction_for_shot
from structure_features import compute_defender_dist_2, compute_free_angle
from advanced_context_features import compute_advanced_context_for_shot
from movement_features import (
    MovementFrame,
    compute_movement_features_for_shot,
    load_movement_frames,
)


# ── Tempo helpers (inlined — avoids importing the tempo script's main) ────────

def _estimate_dt_seconds(curr: FrameState, prev: Optional[FrameState]) -> float:
    if prev is not None and curr.wallclock_ms is not None and prev.wallclock_ms is not None:
        dt = (curr.wallclock_ms - prev.wallclock_ms) / 1000.0
        if dt > 0:
            return dt
    return 0.04  # default: 25 Hz feed


def _speed(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
    dt: float,
) -> float:
    if a is None or b is None or dt <= 0:
        return float("nan")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) / dt)


def _player_xy(
    frame: FrameState,
    player_id: Optional[str],
) -> Optional[Tuple[float, float]]:
    if player_id is None:
        return None
    return frame.home_by_opta.get(player_id) or frame.away_by_opta.get(player_id)


# ── Pressure (inlined to avoid duplicating the full function import) ──────────

def _compute_pressure(
    shooter_opta_id: Optional[str],
    frame: Optional[FrameState],
    radii_m: List[float],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    d_back: float,
    q: float,
    d_front_mode: str,
    d_front_fixed: float,
) -> Dict[str, float]:
    """Pressure features — same logic as compute_pressure_features_for_shot,
    but accepts FrameState (superset of FramePlayers)."""
    out: Dict[str, float] = {}
    for r in radii_m:
        out[f"pressure_def_count_r{_radius_label_m(r)}m"] = float("nan")
    out["pressure_nd_dist_m"] = float("nan")
    out["pressure_defenders_n"] = float("nan")
    out["pressure_total_herold"] = float("nan")

    if frame is None or shooter_opta_id is None:
        return out

    shooter_xy = frame.home_by_opta.get(shooter_opta_id)
    if shooter_xy is not None:
        defenders = frame.away_by_opta
    else:
        shooter_xy = frame.away_by_opta.get(shooter_opta_id)
        if shooter_xy is None:
            return out
        defenders = frame.home_by_opta

    sx, sy = shooter_xy
    if not defenders:
        out["pressure_defenders_n"] = 0.0
        return out

    dists = [math.hypot(x - sx, y - sy) for x, y in defenders.values() if math.isfinite(math.hypot(x - sx, y - sy))]
    if not dists:
        out["pressure_defenders_n"] = 0.0
        return out

    d_arr = np.asarray(dists, dtype=float)
    out["pressure_defenders_n"] = float(d_arr.size)
    out["pressure_nd_dist_m"] = float(np.min(d_arr))

    # DataBallPy Herold kernel pressure
    td_frame, col_id = _to_databallpy_td_frame_and_column_id(frame, shooter_opta_id)
    if td_frame is not None and col_id is not None:
        d_front_arg: str | float = "variable" if d_front_mode == "variable" else float(d_front_fixed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            p_total = get_pressure_on_player(
                td_frame=td_frame,
                column_id=col_id,
                pitch_size=[pitch_length_m, pitch_width_m],
                d_front=d_front_arg,
                d_back=d_back,
                q=q,
            )
        out["pressure_total_herold"] = float(p_total) if pd.notna(p_total) else float("nan")

    for r in radii_m:
        out[f"pressure_def_count_r{_radius_label_m(r)}m"] = float(int(np.sum(d_arr <= r)))

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute all contextual features reading the tracking JSONL exactly once. "
            "Outputs: pressure, goalkeeper, obstruction, structure, tempo, advanced context."
        )
    )
    ap.add_argument("--shots_aligned_csv", type=Path, required=True,
                    help="Aligned shot-level CSV (output of event_tracking_alignment step).")
    ap.add_argument("--tracking_jsonl", type=Path, required=True,
                    help="Second Spectrum tracking JSONL for this match.")
    ap.add_argument("--out_csv", type=Path, required=True,
                    help="Output CSV with all contextual features appended.")
    # Pitch / goal
    ap.add_argument("--pitch_length_m", type=float, default=PITCH_LENGTH_M)
    ap.add_argument("--pitch_width_m", type=float, default=PITCH_WIDTH_M)
    ap.add_argument("--goal_width_m", type=float, default=GOAL_WIDTH_M)
    # Pressure
    ap.add_argument("--radii_m", type=str, default="1,2,3,5",
                    help="Comma-separated defender-proximity radii in metres.")
    ap.add_argument("--d_front_mode", type=str, default="variable",
                    choices=["variable", "fixed"])
    ap.add_argument("--d_back", type=float, default=3.0)
    ap.add_argument("--q", type=float, default=1.75)
    ap.add_argument("--d_front_fixed", type=float, default=10.0)
    # Advanced context
    ap.add_argument("--fast_break_threshold_s", type=float, default=8.0)
    # Movement features
    ap.add_argument("--movement_radius_m", type=float, default=5.0,
                    help="Defender-selection radius for movement features (metres).")
    ap.add_argument("--movement_n_frames", type=int, default=3,
                    help="Number of prior frames used to estimate defender velocity.")
    args = ap.parse_args()

    radii_m = _parse_radii_m(args.radii_m)

    # ── Load shots ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.shots_aligned_csv).copy()
    per_col, frame_col = _resolve_frame_columns(df)
    df[per_col] = pd.to_numeric(df[per_col], errors="coerce").astype("Int64")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").astype("Int64")

    # ── Collect ALL required frame keys (t and t-1 for tempo) ────────────────
    keys: List[FrameKey] = []
    for p, fi in zip(df[per_col].tolist(), df[frame_col].tolist()):
        if pd.isna(p) or pd.isna(fi):
            continue
        per, frame = int(p), int(fi)
        keys.append((per, frame))
        keys.append((per, frame - 1))

    n_unique = len(set(keys))
    print(f"[compute_all_features] Loading {n_unique} unique frames from {args.tracking_jsonl.name} ...")
    frames = load_required_frames_with_state(args.tracking_jsonl, keys)
    print(f"[compute_all_features] Loaded {len(frames)} frames. Computing features for {len(df)} shots ...")

    # ── Load movement frames (t-N … t for velocity window) ───────────────────
    n_frames_mv = args.movement_n_frames
    movement_keys: List[FrameKey] = []
    for p, fi in zip(df[per_col].tolist(), df[frame_col].tolist()):
        if pd.isna(p) or pd.isna(fi):
            continue
        per_i, frame_i = int(p), int(fi)
        for offset in range(n_frames_mv, -1, -1):   # t-N, ..., t-1, t
            movement_keys.append((per_i, frame_i - offset))
    n_mv_unique = len(set(movement_keys))
    print(f"[compute_all_features] Loading {n_mv_unique} movement frames "
          f"(window={n_frames_mv} prior frames) ...")
    mv_frames = load_movement_frames(args.tracking_jsonl, movement_keys)
    print(f"[compute_all_features] Loaded {len(mv_frames)} movement frames.")

    # ── Per-shot computation ──────────────────────────────────────────────────
    pressure_rows: List[Dict[str, float]] = []
    gk_rows: List[Tuple[float, float, float]] = []
    obstruction_vals: List[float] = []
    d2_vals: List[float] = []
    free_vals: List[float] = []
    adv_rows: List[Dict[str, float]] = []
    ball_speeds: List[float] = []
    shooter_speeds: List[float] = []
    movement_rows: List[Dict[str, float]] = []

    nan3 = (float("nan"), float("nan"), float("nan"))

    for _, row in df.iterrows():
        shooter_id = _coerce_opta_id(row.get("player_id"))

        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            # No aligned frame — emit NaNs for every group
            pressure_rows.append({})
            gk_rows.append(nan3)
            obstruction_vals.append(float("nan"))
            d2_vals.append(float("nan"))
            free_vals.append(float("nan"))
            adv_rows.append({"defender_dist_mean3": float("nan")})
            ball_speeds.append(float("nan"))
            shooter_speeds.append(float("nan"))
            movement_rows.append({
                "def_speed_mean_r5m": float("nan"),
                "closing_speed_mean_r5m": float("nan"),
            })
            continue

        key: FrameKey = (int(row[per_col]), int(row[frame_col]))
        curr = frames.get(key)
        prev = frames.get((key[0], key[1] - 1))

        # Pressure
        pressure_rows.append(
            _compute_pressure(
                shooter_opta_id=shooter_id,
                frame=curr,
                radii_m=radii_m,
                pitch_length_m=args.pitch_length_m,
                pitch_width_m=args.pitch_width_m,
                d_back=args.d_back,
                q=args.q,
                d_front_mode=args.d_front_mode,
                d_front_fixed=args.d_front_fixed,
            )
        )

        # Goalkeeper
        gk_rows.append(
            compute_gk_features_for_shot(
                shooter_id=shooter_id,
                frame=curr,  # type: ignore[arg-type]  # FrameState ⊇ FramePlayers
                pitch_length_m=args.pitch_length_m,
                pitch_width_m=args.pitch_width_m,
            )
        )

        # Obstruction
        obstruction_vals.append(
            compute_obstruction_for_shot(
                shooter_id=shooter_id,
                frame=curr,  # type: ignore[arg-type]
                pitch_length_m=args.pitch_length_m,
                pitch_width_m=args.pitch_width_m,
                goal_width_m=args.goal_width_m,
            )
        )

        # Structure (defender_dist_2, free_angle)
        shooter_xy, defenders_xy = resolve_shooter_defenders(shooter_id, curr)  # type: ignore[arg-type]
        if shooter_xy is None or defenders_xy is None:
            d2_vals.append(float("nan"))
            free_vals.append(float("nan"))
        else:
            d2_vals.append(compute_defender_dist_2(shooter_xy, defenders_xy))
            free_vals.append(
                compute_free_angle(
                    shooter_xy,
                    defenders_xy,
                    pitch_length=args.pitch_length_m,
                    pitch_width=args.pitch_width_m,
                )
            )

        # Advanced context (defender_dist_mean3)
        adv_rows.append(
            compute_advanced_context_for_shot(
                shooter_id=shooter_id,
                frame=curr,  # type: ignore[arg-type]
            )
        )

        # Tempo (ball speed, shooter speed via finite difference on frame t-1)
        if curr is None:
            ball_speeds.append(float("nan"))
            shooter_speeds.append(float("nan"))
        else:
            dt = _estimate_dt_seconds(curr, prev)
            ball_prev = prev.ball_xy if prev is not None else None
            ball_speeds.append(_speed(curr.ball_xy, ball_prev, dt))
            s_curr = _player_xy(curr, shooter_id)
            s_prev = _player_xy(prev, shooter_id) if prev is not None else None
            shooter_speeds.append(_speed(s_curr, s_prev, dt))

        # Movement (def_speed_mean_r5m, closing_speed_mean_r5m)
        frame_t_mv = mv_frames.get(key)
        prior_mv = [
            mv_frames.get((key[0], key[1] - k))
            for k in range(n_frames_mv, 0, -1)
        ]
        movement_rows.append(
            compute_movement_features_for_shot(
                shooter_opta_id=shooter_id,
                frame_t=frame_t_mv,
                prior_frames=prior_mv,
                radius_m=args.movement_radius_m,
            )
        )

    # ── Assemble final DataFrame ──────────────────────────────────────────────
    out = df.copy()

    # Pressure columns
    pressure_df = pd.DataFrame(pressure_rows, index=out.index)
    for col in pressure_df.columns:
        out[col] = pressure_df[col]

    # Goalkeeper columns
    out["gk_ball_distance"] = [v[0] for v in gk_rows]
    out["gk_depth"] = [v[1] for v in gk_rows]
    out["gk_lateral_offset"] = [v[2] for v in gk_rows]

    # Obstruction
    out["obstruction_count"] = obstruction_vals

    # Structure
    out["defender_dist_2"] = d2_vals
    out["free_angle"] = free_vals

    # Advanced context
    adv_df = pd.DataFrame(adv_rows, index=out.index)
    for col in adv_df.columns:
        out[col] = adv_df[col]

    # Tempo
    out["ball_speed_mps"] = ball_speeds
    out["shooter_speed_mps"] = shooter_speeds

    # Movement
    movement_df = pd.DataFrame(movement_rows, index=out.index)
    for col in movement_df.columns:
        out[col] = movement_df[col]

    # Sequence-based build-up proxy (possession_length, fast_break)
    if "possession_length" not in out.columns:
        if "possession_duration_s" in out.columns:
            out["possession_length"] = pd.to_numeric(out["possession_duration_s"], errors="coerce")
        elif "time_since_last_event_s" in out.columns:
            out["possession_length"] = pd.to_numeric(out["time_since_last_event_s"], errors="coerce")
        else:
            out["possession_length"] = float("nan")

    pl = pd.to_numeric(out["possession_length"], errors="coerce")
    out["fast_break"] = np.where(pl < args.fast_break_threshold_s, 1.0, 0.0)
    out.loc[pl.isna(), "fast_break"] = np.nan

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[compute_all_features] Done. {len(out)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
