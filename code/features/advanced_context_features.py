from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from frame_utils import (
    FrameKey,
    FramePlayers,
    _coerce_opta_id,
    _resolve_frame_columns,
    load_required_frames,
    resolve_shooter_defenders,
)


def compute_advanced_context_for_shot(
    shooter_id: Optional[str],
    frame: Optional[FramePlayers],
) -> Dict[str, float]:
    out: Dict[str, float] = {"defender_dist_mean3": float("nan")}
    shooter_xy, defenders_xy = resolve_shooter_defenders(shooter_id, frame)
    if shooter_xy is None or defenders_xy is None or len(defenders_xy) == 0:
        return out
    dists = np.linalg.norm(defenders_xy - shooter_xy, axis=1)
    d_sorted = np.sort(dists)
    if len(d_sorted) >= 3:
        out["defender_dist_mean3"] = float(np.mean(d_sorted[:3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute advanced contextual features from tracking freeze-frames.")
    ap.add_argument("--shots_aligned_csv", type=Path, required=True)
    ap.add_argument("--tracking_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--fast_break_threshold_s", type=float, default=8.0)
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

    adv_rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        shooter_id = _coerce_opta_id(row.get("player_id"))
        if pd.isna(row[per_col]) or pd.isna(row[frame_col]):
            adv_rows.append(
                {
                    "defender_dist_mean3": float("nan"),
                }
            )
            continue
        key = (int(row[per_col]), int(row[frame_col]))
        adv_rows.append(
            compute_advanced_context_for_shot(
                shooter_id=shooter_id,
                frame=frames.get(key),
            )
        )

    adv_df = pd.DataFrame(adv_rows)
    out = pd.concat([df.reset_index(drop=True), adv_df.reset_index(drop=True)], axis=1)

    # Sequence-based build-up proxies (event-level fallback when full possession chain is unavailable).
    if "possession_length" not in out.columns:
        if "possession_duration_s" in out.columns:
            out["possession_length"] = pd.to_numeric(out["possession_duration_s"], errors="coerce")
        elif "time_since_last_event_s" in out.columns:
            out["possession_length"] = pd.to_numeric(out["time_since_last_event_s"], errors="coerce")
        else:
            out["possession_length"] = float("nan")

    out["fast_break"] = np.where(
        pd.to_numeric(out["possession_length"], errors="coerce") < float(args.fast_break_threshold_s),
        1.0,
        0.0,
    )
    out.loc[pd.to_numeric(out["possession_length"], errors="coerce").isna(), "fast_break"] = np.nan

    out["__advanced_context_method"] = "freeze_frame_structure_space_plus_build_up_proxy"
    out["__advanced_context_fast_break_threshold_s"] = float(args.fast_break_threshold_s)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("out_csv:", args.out_csv)


if __name__ == "__main__":
    main()

