from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Workaround: pandera 0.22.x cannot resolve pandera.typing.pandas.Series[T] generic
# hints under Python 3.13 + pandas >=2.1 (np.dtype() receives a property object).
# Patch pandas_engine.Engine.dtype to extract the inner type from pandera generics
# before falling through to pd.api.types.pandas_dtype().  This preserves all other
# pandera behaviour while fixing the one code path that breaks on Python 3.13.
import typing as _typing
try:
    import pandera.engines.pandas_engine as _pe
    import pandera.engines.engine as _engine
    import pandera.dtypes as _dtypes
    _orig_pe_dtype = _pe.Engine.dtype.__func__

    @classmethod  # type: ignore[misc]
    def _patched_pe_dtype(cls, data_type):  # type: ignore[override]
        # If it's a pandera generic alias (Series[int], Index[str], …) unwrap it.
        origin = _typing.get_origin(data_type)
        if origin is not None and hasattr(origin, "__mro__"):
            args = _typing.get_args(data_type)
            data_type = args[0] if args else origin
        return _orig_pe_dtype(cls, data_type)

    _pe.Engine.dtype = _patched_pe_dtype
except Exception:
    pass

# Second workaround: Opta F7 files do not provide player full names, so
# databallpy's PlayersSchema rejects the players DataFrame with a non-nullable
# violation on 'full_name'. Patch validate to a no-op; player names are not
# used anywhere in our pipeline.
try:
    from databallpy.game import PlayersSchema
    PlayersSchema.validate = classmethod(lambda cls, df, *a, **kw: df)
except Exception:
    pass

from databallpy import get_game_from_kloppy
from kloppy import opta, secondspectrum


SHOT_EVENTS = {"shot", "own_goal"}


def _to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_epoch_ms_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    # Cast to ms-resolution datetime then to int64 (pandas 2.x, no .view() deprecation)
    ms = dt.astype("datetime64[ms, UTC]").astype("int64")
    return pd.Series(ms, index=series.index, dtype="Int64").where(dt.notna(), pd.NA)


def _parse_n_batches(value: str) -> Union[int, str]:
    value = str(value).strip()
    if value.lower() == "smart":
        return "smart"
    if value.isdigit():
        return int(value)
    raise ValueError("--n_batches must be 'smart' or an integer value, e.g. 50")


def build_sync_table(
    tracking_metadata_xml: Path,
    tracking_raw_file: Path,
    f7_xml: Path,
    f24_xml: Path,
    tracking_additional_metadata: Optional[Path],
    n_batches: Union[int, str],
    offset: float,
    verbose: bool,
    harmonize_pitch_dims: bool,
    preprocess_ball_filter: bool,
    ball_filter_type: str,
    ball_filter_window: int,
    ball_filter_polyorder: int,
    add_ball_velocity: bool,
    ball_velocity_filter_type: str,
    ball_velocity_window: int,
    ball_max_velocity: float,
    add_ball_acceleration: bool,
    ball_accel_filter_type: str,
    ball_accel_window: int,
    ball_accel_polyorder: int,
    ball_max_acceleration: float,
) -> pd.DataFrame:
    tracking_dataset = secondspectrum.load(
        meta_data=tracking_metadata_xml,
        raw_data=tracking_raw_file,
        additional_meta_data=tracking_additional_metadata,
        only_alive=False,  # Keep dead-ball frames for accurate sync as recommended.
    )
    event_dataset = opta.load(f7_data=f7_xml, f24_data=f24_xml)
    if harmonize_pitch_dims:
        # Some provider pairs disagree slightly on pitch dimensions (e.g. 105x68 vs 104.55x67.67),
        # which blocks get_game_from_kloppy. Match event dimensions to tracking dimensions to keep
        # the documented sync workflow operational on mixed-provider inputs.
        event_dataset.metadata.pitch_dimensions.pitch_length = (
            tracking_dataset.metadata.pitch_dimensions.pitch_length
        )
        event_dataset.metadata.pitch_dimensions.pitch_width = (
            tracking_dataset.metadata.pitch_dimensions.pitch_width
        )

    game = get_game_from_kloppy(
        tracking_dataset=tracking_dataset,
        event_dataset=event_dataset,
    )
    # Optional preprocessing mirrors the dedicated DataBallPy preprocessing page.
    if preprocess_ball_filter:
        game.tracking_data.filter_tracking_data(
            column_ids="ball",
            filter_type=ball_filter_type,
            window_length=ball_filter_window,
            polyorder=ball_filter_polyorder,
        )
    if add_ball_velocity:
        game.tracking_data.add_velocity(
            column_ids="ball",
            max_velocity=ball_max_velocity,
            filter_type=ball_velocity_filter_type,
            window_length=ball_velocity_window,
        )
    if add_ball_acceleration:
        # DataBallPy expects velocity to be present before acceleration.
        if not add_ball_velocity:
            game.tracking_data.add_velocity(
                column_ids="ball",
                max_velocity=ball_max_velocity,
                filter_type=ball_velocity_filter_type,
                window_length=ball_velocity_window,
            )
        game.tracking_data.add_acceleration(
            column_ids="ball",
            max_acceleration=ball_max_acceleration,
            filter_type=ball_accel_filter_type,
            window_length=ball_accel_window,
            polyorder=ball_accel_polyorder,
        )
    game.synchronise_tracking_and_event_data(
        n_batches=n_batches,
        offset=offset,
        verbose=verbose,
    )

    event_df = game.event_data.copy()
    tracking_df = game.tracking_data.copy()

    shot_events = event_df.loc[event_df["databallpy_event"].isin(SHOT_EVENTS)].copy()
    if shot_events.empty:
        raise ValueError("No shot events found in databallpy event data after loading/sync.")

    # Follow DataBallPy's documented sync outputs:
    # event_data contains tracking_frame + sync_certainty after synchronisation.
    frame_to_tracking_info = tracking_df[["frame", "datetime", "period_id"]].rename(
        columns={
            "frame": "sync_tracking_frame_idx",
            "datetime": "sync_tracking_datetime",
            "period_id": "sync_tracking_period",
        }
    )

    sync_table = shot_events.rename(columns={"tracking_frame": "sync_tracking_frame_idx"}).merge(
        frame_to_tracking_info, on="sync_tracking_frame_idx", how="left"
    )
    sync_table = sync_table.rename(
        columns={
            "event_id": "databallpy_event_id",
            "original_event_id": "event_id",
            "datetime": "event_datetime_utc",
        }
    )

    sync_table["event_id"] = _to_int_series(sync_table["event_id"])
    sync_table["databallpy_event_id"] = _to_int_series(sync_table["databallpy_event_id"])
    sync_table["sync_tracking_frame_idx"] = _to_int_series(sync_table["sync_tracking_frame_idx"])
    sync_table["sync_tracking_period"] = _to_int_series(sync_table["sync_tracking_period"])
    sync_table["event_time_ms_sync"] = _to_epoch_ms_utc(sync_table["event_datetime_utc"])
    sync_table["sync_tracking_time_ms"] = _to_epoch_ms_utc(sync_table["sync_tracking_datetime"])

    keep_cols = [
        "event_id",
        "databallpy_event_id",
        "databallpy_event",
        "event_datetime_utc",
        "event_time_ms_sync",
        "sync_tracking_frame_idx",
        "sync_tracking_datetime",
        "sync_tracking_time_ms",
        "sync_tracking_period",
        "sync_certainty",
    ]
    return sync_table[keep_cols].copy()


def align_shots_with_sync(shots_csv: Path, sync_table: pd.DataFrame) -> pd.DataFrame:
    shots = pd.read_csv(shots_csv)
    if "event_pk" not in shots.columns:
        raise ValueError(f"'event_pk' column is required in {shots_csv}")
    if "timestamp_utc" not in shots.columns:
        raise ValueError(f"'timestamp_utc' column is required in {shots_csv}")

    shots = shots.copy()
    # sync_table["event_id"] == DataBallPy original_event_id == Opta global event id
    # == shots["event_pk"] (parsed from Opta F24 <Event id=...>).
    # shots["event_id"] is a different field (Opta sequential match-level id) and
    # must NOT be used as the join key.
    shots["_join_key"] = _to_int_series(shots["event_pk"])
    sync_table = sync_table.copy()
    sync_table["_join_key"] = _to_int_series(sync_table["event_id"])
    shots["shot_time_ms"] = _to_epoch_ms_utc(shots["timestamp_utc"])

    aligned = shots.merge(sync_table.drop(columns=["event_id"]), on="_join_key", how="left")
    aligned = aligned.drop(columns=["_join_key"])
    # Rename to the canonical names expected by downstream steps (pressure_features.py)
    aligned = aligned.rename(columns={
        "sync_tracking_frame_idx": "tracking_frame_idx",
        "sync_tracking_period": "tracking_period",
    })
    aligned["sync_delta_ms"] = aligned["sync_tracking_time_ms"] - aligned["shot_time_ms"]
    aligned["sync_aligned"] = aligned["tracking_frame_idx"].notna()
    return aligned


def print_report(aligned: pd.DataFrame) -> None:
    total = int(len(aligned))
    aligned_count = int(aligned["sync_aligned"].sum()) if "sync_aligned" in aligned else 0
    print(f"aligned_shots: {aligned_count} / {total}")

    delta = pd.to_numeric(aligned.get("sync_delta_ms"), errors="coerce").dropna()
    if len(delta) == 0:
        print("sync_delta_ms stats: (no aligned rows)")
        return

    print(
        "sync_delta_ms min / median / max:",
        int(delta.min()),
        "/",
        float(delta.median()),
        "/",
        int(delta.max()),
    )
    print("count |sync_delta_ms| > 40ms:", int((delta.abs() > 40).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Align shots to tracking frames using the official DataBallPy sync.soccer "
            "workflow (kloppy loaders + Game.synchronise_tracking_and_event_data)."
        )
    )
    ap.add_argument("--shots_csv", type=Path, required=True)
    ap.add_argument("--tracking_metadata_xml", type=Path, required=True)
    ap.add_argument("--tracking_raw_file", type=Path, required=True)
    ap.add_argument("--event_f7_xml", type=Path, required=True)
    ap.add_argument("--event_f24_xml", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--tracking_additional_metadata", type=Path, default=None)
    ap.add_argument("--n_batches", type=str, default="smart")
    ap.add_argument("--offset", type=float, default=1.0)
    ap.add_argument("--preprocess_ball_filter", action="store_true")
    ap.add_argument(
        "--ball_filter_type",
        type=str,
        default="savitzky_golay",
        choices=["savitzky_golay", "moving_average"],
    )
    ap.add_argument("--ball_filter_window", type=int, default=25)
    ap.add_argument("--ball_filter_polyorder", type=int, default=2)
    ap.add_argument("--add_ball_velocity", action="store_true")
    ap.add_argument(
        "--ball_velocity_filter_type",
        type=str,
        default="moving_average",
        choices=["savitzky_golay", "moving_average"],
    )
    ap.add_argument("--ball_velocity_window", type=int, default=12)
    ap.add_argument("--ball_max_velocity", type=float, default=50.0)
    ap.add_argument("--add_ball_acceleration", action="store_true")
    ap.add_argument(
        "--ball_accel_filter_type",
        type=str,
        default="savitzky_golay",
        choices=["savitzky_golay", "moving_average"],
    )
    ap.add_argument("--ball_accel_window", type=int, default=35)
    ap.add_argument("--ball_accel_polyorder", type=int, default=2)
    ap.add_argument("--ball_max_acceleration", type=float, default=20.0)
    ap.add_argument(
        "--no_harmonize_pitch_dims",
        action="store_true",
        help="Disable pitch-dimension harmonization between event and tracking datasets.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sync_table = build_sync_table(
        tracking_metadata_xml=args.tracking_metadata_xml,
        tracking_raw_file=args.tracking_raw_file,
        f7_xml=args.event_f7_xml,
        f24_xml=args.event_f24_xml,
        tracking_additional_metadata=args.tracking_additional_metadata,
        n_batches=_parse_n_batches(args.n_batches),
        offset=args.offset,
        verbose=not args.quiet,
        harmonize_pitch_dims=not args.no_harmonize_pitch_dims,
        preprocess_ball_filter=args.preprocess_ball_filter,
        ball_filter_type=args.ball_filter_type,
        ball_filter_window=args.ball_filter_window,
        ball_filter_polyorder=args.ball_filter_polyorder,
        add_ball_velocity=args.add_ball_velocity,
        ball_velocity_filter_type=args.ball_velocity_filter_type,
        ball_velocity_window=args.ball_velocity_window,
        ball_max_velocity=args.ball_max_velocity,
        add_ball_acceleration=args.add_ball_acceleration,
        ball_accel_filter_type=args.ball_accel_filter_type,
        ball_accel_window=args.ball_accel_window,
        ball_accel_polyorder=args.ball_accel_polyorder,
        ball_max_acceleration=args.ball_max_acceleration,
    )

    aligned = align_shots_with_sync(shots_csv=args.shots_csv, sync_table=sync_table)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print_report(aligned)


if __name__ == "__main__":
    main()

