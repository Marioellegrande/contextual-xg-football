from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from shot_geometry import PitchDims, add_classic_geometry_features


SHOT_TYPE_IDS_DEFAULT = (13, 14, 15, 16)  # Opta shot-like events (incl. goal)


def _parse(v: Optional[str], dtype):
    if not v:
        return None
    try:
        return dtype(v)
    except (ValueError, TypeError):
        return None


def parse_pitch_dims_from_metadata_xml(metadata_xml: Path) -> Tuple[int, PitchDims]:
    """
    Read Second Spectrum metadata XML and return (game_id, PitchDims).

    The metadata files in this project contain a single <match ...> element with:
      - iId (match id)
      - fPitchXSizeMeters, fPitchYSizeMeters
    """
    root = ET.parse(metadata_xml).getroot()
    match = root.find(".//match")
    if match is None:
        raise ValueError(f"No <match> element found in metadata XML: {metadata_xml}")

    game_id = _parse(match.attrib.get("iId"), int)
    lx = _parse(match.attrib.get("fPitchXSizeMeters"), float)
    ly = _parse(match.attrib.get("fPitchYSizeMeters"), float)
    if game_id is None or lx is None or ly is None:
        raise ValueError(f"Missing iId/pitch dims in metadata XML: {metadata_xml}")

    return int(game_id), PitchDims(length_m=float(lx), width_m=float(ly))


def infer_game_id_from_eventdetails_filename(eventdetails_xml: Path) -> Optional[int]:
    m = re.search(r"-([0-9]{7})-eventdetails\.xml$", eventdetails_xml.name)
    return int(m.group(1)) if m else None


def iter_shot_events(eventdetails_xml: Path, shot_type_ids: Iterable[int]) -> Iterable[Dict[str, object]]:
    """
    Stream Opta F24 eventdetails.xml and yield dicts for shot events.

    Qualifier IDs extracted:
      15  → header (flag)
      72  → right foot (flag)
      73  → left foot (flag)  [rare — most left foot shots omit 73 and lack 72]
      56  → zone value: "Center", "Left", "Right"
    """
    shot_type_ids_set = {int(x) for x in shot_type_ids}

    # Body part encoding: presence of qualifier flags
    # 15 = head, 72 = right foot, 73 = left foot (absence of both 15+72 → left foot)
    QUAL_HEAD         = 15
    QUAL_RIGHT_FOOT   = 72
    QUAL_LEFT_FOOT    = 73
    QUAL_ZONE         = 56   # value: "Center" / "Left" / "Right"
    QUAL_OPEN_PLAY    = 22   # flag → regular_play
    QUAL_SET_PIECE    = 25   # flag → set piece (free kick)
    QUAL_COUNTER      = 26   # flag → counter_attack
    QUAL_PENALTY      = 23   # flag → penalty
    QUAL_CORNER       = 63   # flag → corner_kick
    QUAL_CROSSED_FK   = 64   # flag → crossed_free_kick

    ctx = ET.iterparse(eventdetails_xml, events=("end",))
    for _ev, elem in ctx:
        if elem.tag != "Event":
            continue

        type_id = _parse(elem.attrib.get("type_id"), int)
        if type_id is None or type_id not in shot_type_ids_set:
            elem.clear()
            continue

        # Read qualifiers
        qual_ids = {int(q.attrib["qualifier_id"]) for q in elem.iter("Q") if "qualifier_id" in q.attrib}
        zone_val = next(
            (q.attrib.get("value", "") for q in elem.iter("Q")
             if q.attrib.get("qualifier_id") == str(QUAL_ZONE)),
            None,
        )

        if QUAL_HEAD in qual_ids:
            body_part = "head"
        elif QUAL_RIGHT_FOOT in qual_ids:
            body_part = "right_foot"
        elif QUAL_LEFT_FOOT in qual_ids:
            body_part = "left_foot"
        else:
            body_part = "foot"  # unspecified foot

        # Derive play_pattern from situation qualifiers
        if QUAL_PENALTY in qual_ids:
            play_pattern = "penalty"
        elif QUAL_CORNER in qual_ids:
            play_pattern = "corner_kick"
        elif QUAL_CROSSED_FK in qual_ids:
            play_pattern = "crossed_free_kick"
        elif QUAL_COUNTER in qual_ids:
            play_pattern = "counter_attack"
        elif QUAL_SET_PIECE in qual_ids:
            play_pattern = "set_piece"
        else:
            play_pattern = "regular_play"  # Q22 or no situation qualifier

        yield {
            "event_pk": _parse(elem.attrib.get("id"), int),
            "event_id": _parse(elem.attrib.get("event_id"), int),
            "type_id": type_id,
            "period_id": _parse(elem.attrib.get("period_id"), int),
            "min": _parse(elem.attrib.get("min"), int),
            "sec": _parse(elem.attrib.get("sec"), int),
            "timestamp_utc": elem.attrib.get("timestamp_utc"),
            "team_id": _parse(elem.attrib.get("team_id"), int),
            "player_id": _parse(elem.attrib.get("player_id"), int),
            "outcome": _parse(elem.attrib.get("outcome"), int),
            "possession_id": None,
            "sequence_id": None,
            "x": _parse(elem.attrib.get("x"), float),
            "y": _parse(elem.attrib.get("y"), float),
            "is_goal": 1 if type_id == 16 else 0,
            "shot_body_part": body_part,
            "shot_zone": zone_val,       # "Center" / "Left" / "Right" / None
            "play_pattern": play_pattern,
        }
        elem.clear()


def _compute_event_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute goal_diff and time_since_last_event_s from the event sequence.
    These are computed post-hoc from the sorted shot table.

    goal_diff: home_goals - away_goals at time of shot (score state before this shot).
    time_since_last_event_s: seconds since previous shot in same match.
    """
    df = df.copy()

    # Shot time in seconds from kick-off (period 1 = 0, period 2 = 45*60, etc.)
    PERIOD_OFFSETS = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
    df["shot_time_s"] = df.apply(
        lambda r: PERIOD_OFFSETS.get(int(r["period_id"]) if pd.notna(r["period_id"]) else 1, 0)
                  + (int(r["min"]) if pd.notna(r["min"]) else 0) * 60
                  + (int(r["sec"]) if pd.notna(r["sec"]) else 0),
        axis=1,
    )

    # time_since_last_event_s: diff from previous shot in same match
    df = df.sort_values(["period_id", "min", "sec", "event_pk"], na_position="last").reset_index(drop=True)
    df["time_since_last_event_s"] = df["shot_time_s"].diff().clip(lower=0)
    df.loc[df.index[0], "time_since_last_event_s"] = float("nan")

    # goal_diff: running score state before each shot
    # We track cumulative goals per team and compute diff at shot time
    # Assumption: team_id of the first shot's team = home team proxy
    # Better: use outcome=1 on type_id=16 as a scored goal
    home_team = df["team_id"].iloc[0]
    home_goals = 0
    away_goals = 0
    goal_diffs = []
    for _, row in df.iterrows():
        goal_diffs.append(home_goals - away_goals)
        if row["is_goal"] == 1:
            if row["team_id"] == home_team:
                home_goals += 1
            else:
                away_goals += 1
    df["goal_diff"] = goal_diffs

    df = df.drop(columns=["shot_time_s"])
    return df


def build_shots_clean_df(
    eventdetails_xml: Path,
    game_id: int,
    pitch: PitchDims,
    shot_type_ids: Iterable[int] = SHOT_TYPE_IDS_DEFAULT,
) -> pd.DataFrame:
    rows = list(iter_shot_events(eventdetails_xml, shot_type_ids=shot_type_ids))
    if not rows:
        raise ValueError(f"No shot events found in {eventdetails_xml}")

    df = pd.DataFrame(rows)
    df.insert(0, "game_id", int(game_id))

    # Compute classic geometry features on the same pitch rectangle as tracking metadata.
    df = add_classic_geometry_features(df, pitch=pitch, x_col="x", y_col="y")

    # Compute event-context features (goal_diff, time_since_last_event_s)
    df = _compute_event_context(df)

    cols = [
        "game_id",
        "event_pk",
        "event_id",
        "type_id",
        "period_id",
        "min",
        "sec",
        "timestamp_utc",
        "team_id",
        "player_id",
        "outcome",
        "possession_id",
        "sequence_id",
        "x",
        "y",
        "distance_m",
        "angle_rad",
        "is_goal",
        "shot_body_part",
        "shot_zone",
        "play_pattern",
        "goal_diff",
        "time_since_last_event_s",
    ]
    df = df[cols].copy()
    df = df.sort_values(["period_id", "min", "sec", "event_pk"], na_position="last").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eventdetails_xml", type=Path, required=True)
    ap.add_argument("--metadata_xml", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument(
        "--shot_type_ids",
        type=int,
        nargs="+",
        default=list(SHOT_TYPE_IDS_DEFAULT),
        help="Opta type_id values to treat as shots (default: 13 14 15 16).",
    )
    args = ap.parse_args()

    inferred_game_id = infer_game_id_from_eventdetails_filename(args.eventdetails_xml)
    meta_game_id, pitch = parse_pitch_dims_from_metadata_xml(args.metadata_xml)

    if inferred_game_id is not None and inferred_game_id != meta_game_id:
        raise ValueError(
            "Game id mismatch between files: "
            f"eventdetails inferred={inferred_game_id}, metadata iId={meta_game_id}"
        )

    shots = build_shots_clean_df(
        eventdetails_xml=args.eventdetails_xml,
        game_id=int(meta_game_id),
        pitch=pitch,
        shot_type_ids=args.shot_type_ids,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print("game_id:", int(meta_game_id))
    print("pitch_length_m:", float(pitch.length_m), "pitch_width_m:", float(pitch.width_m))
    print("n_shots:", int(len(shots)))
    print("n_goals:", int(shots["is_goal"].sum()))


if __name__ == "__main__":
    main()

