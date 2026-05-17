"""
Pressure features from tracking (script)
======================================

Goal
----
Given a shot-level table that is already aligned to tracking (`*_shots_aligned.csv`) and the
corresponding Second Spectrum tracking file (`*_SecondSpectrum_Data.jsonl`), compute a small,
interpretable pressure feature group at the shot moment:

- total pressure using DataBallPy's Adrienko/Herold implementation
- nearest-defender distance (meters)
- defender counts within fixed radii around the shooter

This matches the thesis timeline item:
  "Pressure-gruppe (nearest defender, counts i radii) + validering på 3 kampe."

Method (freeze-frame)
---------------------
For each shot i:
1) Use (tracking_period, tracking_frame_idx) to select exactly one tracking frame (freeze-frame).
2) Identify the shooter position by matching shot `player_id` (Opta id) to tracking `optaId`.
3) Define defenders as the *opponent team* in that same frame.
4) Compute:
   - `pressure_total_herold` via `databallpy.features.get_pressure_on_player(...)`
     (with `d_front="variable"` by default)
   - Euclidean-distance summaries:
   - nd_dist_m = min distance
   - def_count_rXm = count(dist <= rX)

Notes
-----
- Distances are computed in the tracking coordinate system (meters). The coordinate origin does
  not matter for distances.
- The pressure total is delegated to DataBallPy to stay consistent with package behavior.

Example
-------
python code/features/pressure_features.py \\
  --shots_aligned_csv "code/data_pipeline/outputs/shots_aligned/2024-07-21_SIF-SJE_2442547_shots_aligned.csv" \\
  --tracking_jsonl "Data - kampene /2021-07-21 SIF - SJE/20240721-SIF-SJE_613e78b3-8e1f-4170-b1e2-c1965b26821b_SecondSpectrum_Data.jsonl" \\
  --out_csv "code/data_pipeline/outputs/shots_pressure/2024-07-21_SIF-SJE_2442547_shots_pressure.csv" \\
  --radii_m "1,2,3,5"
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from databallpy.features import get_pressure_on_player


FrameKey = Tuple[int, int]  # (period, frameIdx)


def _parse_radii_m(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("radii_m is empty. Example: '1,2,3,5'")
    radii: List[float] = []
    for p in parts:
        try:
            r = float(p)
        except ValueError as e:
            raise ValueError(f"Could not parse radius {p!r} as float.") from e
        if not math.isfinite(r) or r <= 0:
            raise ValueError(f"Radius must be finite and > 0. Got {r}.")
        radii.append(r)
    radii = sorted(set(radii))
    return radii


def _radius_label_m(r: float) -> str:
    """
    Format a radius value as a stable column label.

    Examples:
      1.0  -> "1"
      2.0  -> "2"
      1.5  -> "1p5"
      0.75 -> "0p75"
    """
    if not math.isfinite(r):
        return "nan"
    # treat near-integers as integers
    ri = int(round(r))
    if abs(r - float(ri)) < 1e-9:
        return str(ri)
    # general case: trim trailing zeros, use 'p' as decimal separator
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def _coerce_opta_id(v: object) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        vv = v.strip()
        return vv or None
    # common case: numeric in CSV
    try:
        if pd.isna(v):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    try:
        # avoid "490511.0"
        return str(int(float(v)))
    except Exception:
        return str(v)


@dataclass(frozen=True)
class FramePlayers:
    home_by_opta: Dict[str, Tuple[float, float]]
    away_by_opta: Dict[str, Tuple[float, float]]


def _extract_players_xy(players: object) -> Dict[str, Tuple[float, float]]:
    """
    Convert tracking players list → mapping: optaId -> (x, y) in meters.
    Ignores missing/invalid rows.
    """
    if not isinstance(players, list):
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        opta = _coerce_opta_id(p.get("optaId"))
        xyz = p.get("xyz")
        if opta is None or not isinstance(xyz, list) or len(xyz) < 2:
            continue
        try:
            x = float(xyz[0])
            y = float(xyz[1])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out[opta] = (x, y)
    return out


def load_required_frames(
    tracking_jsonl: Path,
    required_keys: Iterable[FrameKey],
) -> Dict[FrameKey, FramePlayers]:
    """
    Scan tracking jsonl once and keep only frames needed for the given shots.
    Keys are (period, frameIdx) to avoid collisions across halves.
    """
    required = set(required_keys)
    frames: Dict[FrameKey, FramePlayers] = {}

    bad_json = 0
    missing_fields = 0

    with tracking_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            per = obj.get("period")
            fi = obj.get("frameIdx")
            if per is None or fi is None:
                missing_fields += 1
                continue

            try:
                key: FrameKey = (int(per), int(fi))
            except Exception:
                missing_fields += 1
                continue

            if key not in required:
                continue

            home = _extract_players_xy(obj.get("homePlayers"))
            away = _extract_players_xy(obj.get("awayPlayers"))
            frames[key] = FramePlayers(home_by_opta=home, away_by_opta=away)

            # small optimization: stop early
            if len(frames) >= len(required):
                break

    if not frames:
        raise ValueError(
            f"No required frames found in tracking file {tracking_jsonl}. "
            f"bad_json={bad_json}, missing_fields={missing_fields}"
        )
    return frames


def _to_databallpy_td_frame_and_column_id(
    frame: FramePlayers, shooter_opta_id: str
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Build a synthetic DataBallPy-like td_frame for one freeze-frame.
    Column naming follows DataBallPy convention: home_<id>_x / away_<id>_x.
    """
    shooter_in_home = shooter_opta_id in frame.home_by_opta
    shooter_in_away = shooter_opta_id in frame.away_by_opta
    if not shooter_in_home and not shooter_in_away:
        return None, None

    values: Dict[str, float] = {}
    for pid, (x, y) in frame.home_by_opta.items():
        values[f"home_{pid}_x"] = float(x)
        values[f"home_{pid}_y"] = float(y)
    for pid, (x, y) in frame.away_by_opta.items():
        values[f"away_{pid}_x"] = float(x)
        values[f"away_{pid}_y"] = float(y)

    col_id = f"home_{shooter_opta_id}" if shooter_in_home else f"away_{shooter_opta_id}"
    return pd.Series(values), col_id


def compute_pressure_features_for_shot(
    shooter_opta_id: Optional[str],
    frame: Optional[FramePlayers],
    radii_m: Sequence[float],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    d_back: float,
    q: float,
    d_front_mode: str,
    d_front_fixed: float,
) -> Dict[str, float]:
    """
    Compute nearest-defender distance + counts-in-radii for one shot.
    Returns NaNs when shooter/frame info is missing.
    """
    out: Dict[str, float] = {}

    for r in radii_m:
        out[f"pressure_def_count_r{_radius_label_m(float(r))}m"] = float("nan")
    out["pressure_nd_dist_m"] = float("nan")
    out["pressure_defenders_n"] = float("nan")
    out["pressure_total_herold"] = float("nan")

    if frame is None or shooter_opta_id is None:
        return out

    shooter_xy = frame.home_by_opta.get(shooter_opta_id)
    defenders: Dict[str, Tuple[float, float]]
    team = "home"
    if shooter_xy is not None:
        defenders = frame.away_by_opta
    else:
        shooter_xy = frame.away_by_opta.get(shooter_opta_id)
        if shooter_xy is None:
            return out
        team = "away"
        defenders = frame.home_by_opta

    sx, sy = shooter_xy
    if not defenders:
        out["pressure_defenders_n"] = 0.0
        return out

    dists: List[float] = []
    for xy in defenders.values():
        dx, dy = xy
        d = math.hypot(dx - sx, dy - sy)
        if math.isfinite(d):
            dists.append(d)

    if not dists:
        out["pressure_defenders_n"] = 0.0
        return out

    d_arr = np.asarray(dists, dtype=float)
    out["pressure_defenders_n"] = float(d_arr.size)
    out["pressure_nd_dist_m"] = float(np.min(d_arr))
    td_frame, col_id = _to_databallpy_td_frame_and_column_id(frame, shooter_opta_id)
    if td_frame is None or col_id is None:
        return out
    d_front_arg: str | float = "variable" if d_front_mode == "variable" else float(d_front_fixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        pressure_total = get_pressure_on_player(
            td_frame=td_frame,
            column_id=col_id,
            pitch_size=[float(pitch_length_m), float(pitch_width_m)],
            d_front=d_front_arg,
            d_back=float(d_back),
            q=float(q),
        )
    out["pressure_total_herold"] = float(pressure_total) if pd.notna(pressure_total) else float("nan")

    for r in radii_m:
        cnt = int(np.sum(d_arr <= float(r)))
        out[f"pressure_def_count_r{_radius_label_m(float(r))}m"] = float(cnt)

    return out


def add_pressure_features(
    shots_aligned: pd.DataFrame,
    frames: Dict[FrameKey, FramePlayers],
    radii_m: Sequence[float],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    d_back: float,
    q: float,
    d_front_mode: str,
    d_front_fixed: float,
) -> pd.DataFrame:
    required_cols = ["player_id", "tracking_frame_idx", "tracking_period"]
    missing = [c for c in required_cols if c not in shots_aligned.columns]
    if missing:
        raise ValueError(f"shots_aligned missing required columns: {missing}")

    df = shots_aligned.copy()

    # ensure keys are ints where possible
    df["tracking_period"] = pd.to_numeric(df["tracking_period"], errors="coerce").astype("Int64")
    df["tracking_frame_idx"] = pd.to_numeric(df["tracking_frame_idx"], errors="coerce").astype("Int64")

    # compute per-row
    feats_rows: List[Dict[str, float]] = []
    n_missing_frame = 0
    n_missing_shooter = 0
    for _, row in df.iterrows():
        shooter_id = _coerce_opta_id(row.get("player_id"))
        if shooter_id is None:
            n_missing_shooter += 1
        per = row.get("tracking_period")
        fi = row.get("tracking_frame_idx")
        key: Optional[FrameKey] = None
        if per is not None and fi is not None and not (pd.isna(per) or pd.isna(fi)):
            try:
                key = (int(per), int(fi))
            except Exception:
                key = None

        frame = frames.get(key) if key is not None else None
        if key is not None and frame is None:
            n_missing_frame += 1

        feats = compute_pressure_features_for_shot(
            shooter_id,
            frame,
            radii_m=radii_m,
            pitch_length_m=float(pitch_length_m),
            pitch_width_m=float(pitch_width_m),
            d_back=float(d_back),
            q=float(q),
            d_front_mode=str(d_front_mode),
            d_front_fixed=float(d_front_fixed),
        )
        feats_rows.append(feats)

    feats_df = pd.DataFrame(feats_rows)
    out = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # helpful run metadata
    out["__pressure_radii_m"] = ",".join(_radius_label_m(float(r)) for r in radii_m)
    out["__pressure_method"] = "databallpy_get_pressure_on_player"
    out["__pressure_d_front_mode"] = str(d_front_mode)
    out["__pressure_d_back"] = float(d_back)
    out["__pressure_q"] = float(q)
    out["__pressure_pitch_length_m"] = float(pitch_length_m)
    out["__pressure_pitch_width_m"] = float(pitch_width_m)
    if d_front_mode == "fixed":
        out["__pressure_d_front_fixed"] = float(d_front_fixed)
    else:
        out["__pressure_d_front_fixed"] = pd.NA
    out["__pressure_missing_shooter_rows"] = n_missing_shooter
    out["__pressure_missing_frame_rows"] = n_missing_frame
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute tracking-derived pressure features for aligned shots "
            "(DataBallPy Adrienko/Herold pressure + proximity summaries)."
        )
    )
    ap.add_argument("--shots_aligned_csv", type=Path, required=True)
    ap.add_argument("--tracking_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--radii_m", type=str, default="1,2,3,5", help="Comma-separated radii in meters.")
    ap.add_argument(
        "--d_front_mode",
        type=str,
        default="variable",
        choices=["variable", "fixed"],
        help="Pressure model front distance mode (Herold='variable', Andrienko='fixed').",
    )
    ap.add_argument("--d_front_fixed", type=float, default=9.0, help="Fixed d_front when --d_front_mode=fixed.")
    ap.add_argument("--d_back", type=float, default=3.0, help="Back distance in pressure model.")
    ap.add_argument("--q", type=float, default=1.75, help="Exponent q in pressure model.")
    ap.add_argument("--pitch_length_m", type=float, default=105.0, help="Pitch length in meters.")
    ap.add_argument("--pitch_width_m", type=float, default=68.0, help="Pitch width in meters.")
    args = ap.parse_args()

    radii_m = _parse_radii_m(args.radii_m)

    shots = pd.read_csv(args.shots_aligned_csv)
    period = pd.to_numeric(shots["tracking_period"], errors="coerce")
    frame_idx = pd.to_numeric(shots["tracking_frame_idx"], errors="coerce")
    valid_mask = period.notna() & frame_idx.notna()
    keys: List[FrameKey] = list(zip(period[valid_mask].astype(int), frame_idx[valid_mask].astype(int)))
    if not keys:
        raise ValueError("No valid (tracking_period, tracking_frame_idx) keys found in shots_aligned_csv.")

    frames = load_required_frames(args.tracking_jsonl, required_keys=keys)
    out = add_pressure_features(
        shots,
        frames=frames,
        radii_m=radii_m,
        pitch_length_m=float(args.pitch_length_m),
        pitch_width_m=float(args.pitch_width_m),
        d_back=float(args.d_back),
        q=float(args.q),
        d_front_mode=str(args.d_front_mode),
        d_front_fixed=float(args.d_front_fixed),
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # basic summary for quick validation
    n = len(out)
    n_nan = int(pd.isna(out["pressure_nd_dist_m"]).sum())
    print("n_shots:", n)
    print("n_missing_pressure_rows:", n_nan)
    if "pressure_total_herold" in out.columns:
        print("mean_pressure_total_herold:", float(pd.to_numeric(out["pressure_total_herold"], errors="coerce").mean()))
    if "is_goal" in out.columns:
        g = pd.to_numeric(out["is_goal"], errors="coerce")
        nd = pd.to_numeric(out["pressure_nd_dist_m"], errors="coerce")
        herold = pd.to_numeric(out["pressure_total_herold"], errors="coerce")
        print("mean_nd_dist_m_goal:", float(nd[g == 1].mean()))
        print("mean_nd_dist_m_nongoal:", float(nd[g == 0].mean()))
        print("mean_pressure_total_herold_goal:", float(herold[g == 1].mean()))
        print("mean_pressure_total_herold_nongoal:", float(herold[g == 0].mean()))
    print("out_csv:", str(args.out_csv))


if __name__ == "__main__":
    main()

