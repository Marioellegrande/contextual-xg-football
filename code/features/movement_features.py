"""Defender movement features from tracking data.
=================================================

Two features computed at the shot freeze-frame:

def_speed_mean_r5m
    Mean scalar speed (m/s) of defenders within ``radius_m`` metres of
    the shooter, taken directly from the tracking ``speed`` field.

closing_speed_mean_r5m
    Mean closing speed (m/s) of defenders within ``radius_m`` metres of
    the shooter.  Closing speed is the projection of the defender's velocity
    vector onto the defender→ball unit vector (positive = moving toward ball,
    negative = retreating).  Velocity is estimated from positional differences
    over the last N frames (default N=3, Δt = 0.04 s at 25 Hz).

Both features fall back to the nearest defender when no defender is within
the radius, and return NaN when frame data or ball position is missing.

Feature definitions
-------------------
Let d be a defender within ``radius_m`` metres of the shooter at frame t.

def_speed_mean_r5m:
    s_d = speed field from tracking data (m/s)
    def_speed_mean_r5m = mean(s_d) over nearby defenders

closing_speed_mean_r5m:
    positions p_k = (x_k, y_k) at frames t-N, ..., t
    velocity: vx = mean(Δx / Δt), vy = mean(Δy / Δt)  where Δt = 0.04 s
    unit vector toward ball: u = (ball_xy - d_xy) / ||ball_xy - d_xy||
    closing_speed = vx·u_x + vy·u_y
    closing_speed_mean_r5m = mean(closing_speed) over nearby defenders
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from frame_utils import FrameKey, _coerce_opta_id, _extract_ball_xy

DT_DEFAULT: float = 0.04  # seconds between frames (25 Hz Second Spectrum feed)


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MovementFrame:
    """Tracking frame snapshot for movement feature computation.

    Extends the position-only FrameState with per-player scalar speed,
    which is provided directly by the Second Spectrum JSONL feed.
    """

    period: int
    frame_idx: int
    home_xy: Dict[str, Tuple[float, float]]   # optaId -> (x, y)  in metres
    away_xy: Dict[str, Tuple[float, float]]
    home_speed: Dict[str, float]              # optaId -> speed  (m/s)
    away_speed: Dict[str, float]
    ball_xy: Optional[Tuple[float, float]]    # None when ball tracking is lost


# ── Frame loading ──────────────────────────────────────────────────────────────

def _extract_players_xy_and_speed(
    players: object,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """Parse a tracking players list into position and speed maps.

    Returns
    -------
    xy_map    : optaId -> (x, y)
    speed_map : optaId -> speed (m/s)
    """
    if not isinstance(players, list):
        return {}, {}
    xy_map: Dict[str, Tuple[float, float]] = {}
    speed_map: Dict[str, float] = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = _coerce_opta_id(p.get("optaId"))
        xyz = p.get("xyz")
        if pid is None or not isinstance(xyz, list) or len(xyz) < 2:
            continue
        try:
            x, y = float(xyz[0]), float(xyz[1])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        xy_map[pid] = (x, y)
        try:
            s = float(p.get("speed", float("nan")))
            if math.isfinite(s):
                speed_map[pid] = s
        except Exception:
            pass
    return xy_map, speed_map


def load_movement_frames(
    tracking_jsonl: Path,
    required_keys: Iterable[FrameKey],
) -> Dict[FrameKey, MovementFrame]:
    """Scan the tracking JSONL once and return MovementFrame objects.

    Captures (x, y) positions, scalar speed, and ball position for every
    requested (period, frameIdx) key.  Keys not found in the file are
    silently absent from the returned dict.
    """
    required = set(required_keys)
    frames: Dict[FrameKey, MovementFrame] = {}

    with tracking_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                per = int(obj["period"])
                fi = int(obj["frameIdx"])
            except Exception:
                continue
            key = (per, fi)
            if key not in required:
                continue
            home_xy, home_speed = _extract_players_xy_and_speed(
                obj.get("homePlayers")
            )
            away_xy, away_speed = _extract_players_xy_and_speed(
                obj.get("awayPlayers")
            )
            frames[key] = MovementFrame(
                period=per,
                frame_idx=fi,
                home_xy=home_xy,
                away_xy=away_xy,
                home_speed=home_speed,
                away_speed=away_speed,
                ball_xy=_extract_ball_xy(obj.get("ball")),
            )
            if len(frames) >= len(required):
                break

    return frames


# ── Feature computation ────────────────────────────────────────────────────────

def compute_movement_features_for_shot(
    shooter_opta_id: Optional[str],
    frame_t: Optional[MovementFrame],
    prior_frames: List[Optional[MovementFrame]],
    *,
    radius_m: float = 5.0,
) -> Dict[str, float]:
    """Compute def_speed_mean_r5m and closing_speed_mean_r5m for one shot.

    Parameters
    ----------
    shooter_opta_id
        The shooter's Opta player ID string.
    frame_t
        Freeze-frame at the shot moment (frame t).
    prior_frames
        List of frames [t-N, ..., t-1] in chronological order (oldest first).
        Used to estimate defender velocity via finite differences.
    radius_m
        Defender-selection radius around the shooter in metres (default 5.0).

    Returns
    -------
    dict with keys ``def_speed_mean_r5m`` and ``closing_speed_mean_r5m``.
    Both values are NaN when the shooter position or frame data is unavailable.

    Notes
    -----
    Positive closing_speed means the defender is moving toward the ball;
    negative means retreating.  A value of 0 means lateral movement.
    """
    nan_out: Dict[str, float] = {
        "def_speed_mean_r5m": float("nan"),
        "closing_speed_mean_r5m": float("nan"),
    }

    if frame_t is None or shooter_opta_id is None:
        return nan_out

    # ── Identify shooter side; set defenders as the opposing team ────────────
    shooter_xy = frame_t.home_xy.get(shooter_opta_id)
    if shooter_xy is not None:
        defenders_xy = frame_t.away_xy
        defenders_speed = frame_t.away_speed
        prior_def_xy: List[Dict[str, Tuple[float, float]]] = [
            (f.away_xy if f is not None else {}) for f in prior_frames
        ]
    else:
        shooter_xy = frame_t.away_xy.get(shooter_opta_id)
        if shooter_xy is None:
            return nan_out
        defenders_xy = frame_t.home_xy
        defenders_speed = frame_t.home_speed
        prior_def_xy = [
            (f.home_xy if f is not None else {}) for f in prior_frames
        ]

    sx, sy = shooter_xy

    # ── Select defenders within radius; fall back to nearest if none ─────────
    nearby: List[str] = [
        pid
        for pid, (dx, dy) in defenders_xy.items()
        if math.hypot(dx - sx, dy - sy) <= radius_m
    ]
    if not nearby:
        if not defenders_xy:
            return nan_out
        nearby = [
            min(
                defenders_xy,
                key=lambda pid: math.hypot(
                    defenders_xy[pid][0] - sx, defenders_xy[pid][1] - sy
                ),
            )
        ]

    ball_xy = frame_t.ball_xy  # may be None

    def_speed_vals: List[float] = []
    closing_speed_vals: List[float] = []

    for pid in nearby:

        # ── Feature 1: speed from tracking data (direct) ─────────────────
        spd = defenders_speed.get(pid)
        if spd is not None and math.isfinite(spd):
            def_speed_vals.append(spd)

        # ── Feature 2: closing speed (velocity projected onto ball dir) ──
        if ball_xy is None:
            # Ball position unavailable — cannot compute closing direction
            continue

        # Build ordered position list: prior frames → current frame
        positions: List[Tuple[float, float]] = []
        for pf_dict in prior_def_xy:
            pos = pf_dict.get(pid)
            if pos is not None:
                positions.append(pos)
        curr_pos = defenders_xy.get(pid)
        if curr_pos is not None:
            positions.append(curr_pos)

        if len(positions) >= 2:
            # Finite difference over available consecutive frames
            deltas_x = [
                (positions[i + 1][0] - positions[i][0]) / DT_DEFAULT
                for i in range(len(positions) - 1)
            ]
            deltas_y = [
                (positions[i + 1][1] - positions[i][1]) / DT_DEFAULT
                for i in range(len(positions) - 1)
            ]
            vx = float(np.mean(deltas_x))
            vy = float(np.mean(deltas_y))
        else:
            # Fewer than 2 positions (very start of period) — assume stationary
            vx, vy = 0.0, 0.0

        # Unit vector: defender → ball
        dx_def, dy_def = defenders_xy[pid]
        bx, by = ball_xy
        diff_x = bx - dx_def
        diff_y = by - dy_def
        norm = math.hypot(diff_x, diff_y) + 1e-6  # guard against zero norm
        ux, uy = diff_x / norm, diff_y / norm

        # Projection of velocity onto the closing direction
        closing_speed_vals.append(vx * ux + vy * uy)

    finite_speeds = [v for v in def_speed_vals if math.isfinite(v)]
    finite_closing = [v for v in closing_speed_vals if math.isfinite(v)]

    return {
        "def_speed_mean_r5m": (
            float(np.mean(finite_speeds)) if finite_speeds else float("nan")
        ),
        "closing_speed_mean_r5m": (
            float(np.mean(finite_closing)) if finite_closing else float("nan")
        ),
    }
