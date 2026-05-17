"""Shared tracking-frame utilities used by all feature extraction scripts."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ── Pitch & goal constants (standard professional football) ───────────────────
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0
GOAL_WIDTH_M: float = 7.32


FrameKey = Tuple[int, int]  # (period, frameIdx)


def _coerce_opta_id(v: object) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    try:
        if pd.isna(v):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)


@dataclass(frozen=True)
class FramePlayers:
    home_by_opta: Dict[str, Tuple[float, float]]
    away_by_opta: Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class FrameState:
    period: int
    frame_idx: int
    wallclock_ms: Optional[int]
    ball_xy: Optional[Tuple[float, float]]
    home_by_opta: Dict[str, Tuple[float, float]]
    away_by_opta: Dict[str, Tuple[float, float]]


def _extract_players_xy(players: object) -> Dict[str, Tuple[float, float]]:
    if not isinstance(players, list):
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = _coerce_opta_id(p.get("optaId"))
        xyz = p.get("xyz")
        if pid is None or not isinstance(xyz, list) or len(xyz) < 2:
            continue
        try:
            x = float(xyz[0])
            y = float(xyz[1])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            out[pid] = (x, y)
    return out


def _extract_ball_xy(ball_obj: object) -> Optional[Tuple[float, float]]:
    if isinstance(ball_obj, dict):
        xyz = ball_obj.get("xyz")
        if isinstance(xyz, list) and len(xyz) >= 2:
            try:
                x, y = float(xyz[0]), float(xyz[1])
                if math.isfinite(x) and math.isfinite(y):
                    return (x, y)
            except Exception:
                pass
    return None


def _extract_wallclock_ms(obj: dict) -> Optional[int]:
    wc = obj.get("wallClock")
    try:
        return None if wc is None else int(wc)
    except Exception:
        return None


def _resolve_frame_columns(df: pd.DataFrame) -> Tuple[str, str]:
    period_candidates = ["tracking_period", "sync_tracking_period"]
    frame_candidates = ["tracking_frame_idx", "sync_tracking_frame_idx", "tracking_idx"]
    per_col = next((c for c in period_candidates if c in df.columns), None)
    frame_col = next((c for c in frame_candidates if c in df.columns), None)
    if per_col is None or frame_col is None:
        raise ValueError(
            "Could not resolve frame columns. Need one of "
            f"{period_candidates} and one of {frame_candidates}."
        )
    return per_col, frame_col


def load_required_frames(
    tracking_jsonl: Path,
    required_keys: Iterable[FrameKey],
) -> Dict[FrameKey, FramePlayers]:
    """Scan tracking JSONL once and return only the frames needed for the given shots."""
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
            frames[key] = FramePlayers(
                home_by_opta=_extract_players_xy(obj.get("homePlayers")),
                away_by_opta=_extract_players_xy(obj.get("awayPlayers")),
            )
            if len(frames) >= len(required):
                break
    if not frames:
        raise ValueError(
            f"No required frames found in {tracking_jsonl}. "
            f"bad_json={bad_json}, missing_fields={missing_fields}"
        )
    return frames


def resolve_shooter_defenders(
    shooter_id: Optional[str],
    frame: Optional[FramePlayers],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (shooter_xy, defenders_xy) for a shot, or (None, None) if shooter not found.

    Determines shooter's team by checking home then away lookup, and returns
    the opposing team as defenders.  Both outputs are float64 numpy arrays:
      - shooter_xy: shape (2,)
      - defenders_xy: shape (N, 2), may be empty (N=0) when no defenders found
    """
    if shooter_id is None or frame is None:
        return None, None
    shooter_pos = frame.home_by_opta.get(shooter_id)
    if shooter_pos is not None:
        defenders = frame.away_by_opta
    else:
        shooter_pos = frame.away_by_opta.get(shooter_id)
        if shooter_pos is None:
            return None, None
        defenders = frame.home_by_opta
    shooter_xy = np.asarray(shooter_pos, dtype=float)
    if not defenders:
        return shooter_xy, np.empty((0, 2), dtype=float)
    defenders_xy = np.asarray(list(defenders.values()), dtype=float)
    if defenders_xy.ndim != 2 or defenders_xy.shape[1] != 2:
        return shooter_xy, np.empty((0, 2), dtype=float)
    return shooter_xy, defenders_xy


def load_required_frames_with_state(
    tracking_jsonl: Path,
    required_keys: Iterable[FrameKey],
) -> Dict[FrameKey, FrameState]:
    """Like load_required_frames but also captures ball position and wallclock time."""
    required = set(required_keys)
    frames: Dict[FrameKey, FrameState] = {}
    with tracking_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                per = int(obj.get("period"))
                fi = int(obj.get("frameIdx"))
            except Exception:
                continue
            key = (per, fi)
            if key not in required:
                continue
            frames[key] = FrameState(
                period=per,
                frame_idx=fi,
                wallclock_ms=_extract_wallclock_ms(obj),
                ball_xy=_extract_ball_xy(obj.get("ball")),
                home_by_opta=_extract_players_xy(obj.get("homePlayers")),
                away_by_opta=_extract_players_xy(obj.get("awayPlayers")),
            )
            if len(frames) >= len(required):
                break
    return frames
