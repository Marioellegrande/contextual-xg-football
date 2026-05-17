from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _run_step(command: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object.")
    return data


def _find_match_bundles(data_root: Path) -> list[dict[str, Path]]:
    bundles: list[dict[str, Path]] = []
    for outer in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        inner_dirs = [p for p in outer.iterdir() if p.is_dir() and not p.name.startswith("__")]
        if not inner_dirs:
            continue
        inner = inner_dirs[0]
        f24 = next(inner.glob("f24-*-eventdetails.xml"), None)
        f7 = next(inner.glob("srml-*-matchresults.xml"), None)
        meta = next(inner.glob("*_SecondSpectrum_Metadata.xml"), None)
        raw = next(inner.glob("*_SecondSpectrum_Data.jsonl"), None)
        if all([f24, f7, meta, raw]):
            bundles.append(
                {
                    "match_name": outer.name,
                    "event_f24_xml": f24,
                    "event_f7_xml": f7,
                    "tracking_metadata_xml": meta,
                    "tracking_raw_file": raw,
                }
            )
    return bundles


def _extract_game_id_from_f24(path: Path) -> str:
    m = re.search(r"-(\d+)-eventdetails\.xml$", path.name)
    if not m:
        raise ValueError(f"Could not parse game_id from {path.name}")
    return m.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contextual xG pipeline for all matches under a data root.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/raw/Superliga 2024-2025 (Test Data)"),
        help="Root containing one folder per match bundle.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    python_bin = sys.executable

    outputs_cfg = cfg.get("outputs", {})
    align = cfg.get("alignment", {})
    pressure = cfg.get("pressure", {})
    structure = cfg.get("structure", {})
    advanced_context = cfg.get("advanced_context", {})
    obstruction = cfg.get("obstruction", {})
    goalkeeper = cfg.get("goalkeeper", {})
    modeling = cfg.get("modeling", {})
    pipeline_cfg = cfg.get("pipeline", {})
    feature_toggles = pipeline_cfg.get("features", {})

    bundles = _find_match_bundles(args.data_root)
    if not bundles:
        raise ValueError(f"No match bundles found in {args.data_root}")

    out_root = project_root / "code" / "data_pipeline" / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    xg_out_root = project_root / "code" / "xg_baseline" / "outputs" / "classic_xg"
    xg_out_root.mkdir(parents=True, exist_ok=True)

    features_final_paths: list[Path] = []
    for bundle in bundles:
        game_id = _extract_game_id_from_f24(bundle["event_f24_xml"])
        print(f"\n=== Running match {bundle['match_name']} (game_id={game_id}) ===")

        shots_clean_csv = out_root / f"{game_id}_shots_clean.csv"
        shots_aligned_csv = out_root / f"{game_id}_shots_aligned.csv"
        shots_pressure_csv = out_root / f"{game_id}_shots_pressure.csv"
        shots_structure_csv = out_root / f"{game_id}_shots_structure.csv"
        shots_advanced_context_csv = out_root / f"{game_id}_shots_advanced_context.csv"
        shots_obstruction_csv = out_root / f"{game_id}_shots_obstruction.csv"
        shots_goalkeeper_csv = out_root / f"{game_id}_shots_goalkeeper.csv"
        shots_tempo_csv = out_root / f"{game_id}_shots_tempo.csv"
        features_final_csv = out_root / f"{game_id}_features_final.csv"
        classic_xg_csv = xg_out_root / f"{game_id}_classic_xg.csv"

        _run_step(
            [
                python_bin,
                str(project_root / "code" / "data_pipeline" / "extract_shots_clean_from_eventdetails.py"),
                "--eventdetails_xml",
                str(bundle["event_f24_xml"]),
                "--metadata_xml",
                str(bundle["tracking_metadata_xml"]),
                "--out_csv",
                str(shots_clean_csv),
            ],
            dry_run=args.dry_run,
        )
        _run_step(
            [
                python_bin,
                str(project_root / "code" / "data_pipeline" / "event_tracking_alignment_databallpy.py"),
                "--shots_csv",
                str(shots_clean_csv),
                "--tracking_metadata_xml",
                str(bundle["tracking_metadata_xml"]),
                "--tracking_raw_file",
                str(bundle["tracking_raw_file"]),
                "--event_f7_xml",
                str(bundle["event_f7_xml"]),
                "--event_f24_xml",
                str(bundle["event_f24_xml"]),
                "--n_batches",
                str(align.get("n_batches", "smart")),
                "--offset",
                str(align.get("offset", 1.0)),
                "--out_csv",
                str(shots_aligned_csv),
            ],
            dry_run=args.dry_run,
        )

        feature_input_csv = shots_aligned_csv
        if bool(feature_toggles.get("pressure", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "pressure_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--d_front_mode",
                    str(pressure.get("d_front_mode", "variable")),
                    "--d_back",
                    str(pressure.get("d_back", 3.0)),
                    "--q",
                    str(pressure.get("q", 1.75)),
                    "--pitch_length_m",
                    str(pressure.get("pitch_length_m", 105.0)),
                    "--pitch_width_m",
                    str(pressure.get("pitch_width_m", 68.0)),
                    "--out_csv",
                    str(shots_pressure_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_pressure_csv

        if bool(feature_toggles.get("structure", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "structure_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--pitch_length_m",
                    str(structure.get("pitch_length_m", 105.0)),
                    "--pitch_width_m",
                    str(structure.get("pitch_width_m", 68.0)),
                    "--out_csv",
                    str(shots_structure_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_structure_csv

        if bool(feature_toggles.get("advanced_context", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "advanced_context_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--pitch_length_m",
                    str(advanced_context.get("pitch_length_m", 105.0)),
                    "--goal_width_m",
                    str(advanced_context.get("goal_width_m", 7.32)),
                    "--fast_break_threshold_s",
                    str(advanced_context.get("fast_break_threshold_s", 8.0)),
                    "--out_csv",
                    str(shots_advanced_context_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_advanced_context_csv

        if bool(feature_toggles.get("obstruction", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "obstruction_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--pitch_length_m",
                    str(obstruction.get("pitch_length_m", 105.0)),
                    "--goal_width_m",
                    str(obstruction.get("goal_width_m", 7.32)),
                    "--out_csv",
                    str(shots_obstruction_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_obstruction_csv

        if bool(feature_toggles.get("goalkeeper", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "goalkeeper_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--pitch_length_m",
                    str(goalkeeper.get("pitch_length_m", 105.0)),
                    "--out_csv",
                    str(shots_goalkeeper_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_goalkeeper_csv

        if bool(feature_toggles.get("tempo", True)):
            _run_step(
                [
                    python_bin,
                    str(project_root / "code" / "features" / "tempo_features.py"),
                    "--shots_aligned_csv",
                    str(feature_input_csv),
                    "--tracking_jsonl",
                    str(bundle["tracking_raw_file"]),
                    "--out_csv",
                    str(shots_tempo_csv),
                ],
                dry_run=args.dry_run,
            )
            feature_input_csv = shots_tempo_csv

        _run_step(
            [
                python_bin,
                "-c",
                "import pandas as pd,sys; df=pd.read_csv(sys.argv[1]); df.to_csv(sys.argv[2], index=False)",
                str(feature_input_csv),
                str(features_final_csv),
            ],
            dry_run=args.dry_run,
        )
        features_final_paths.append(features_final_csv)

        _run_step(
            [
                python_bin,
                str(project_root / "code" / "xg_baseline" / "classic_xg.py"),
                "--inputs",
                str(features_final_csv),
                "--out_csv",
                str(classic_xg_csv),
            ],
            dry_run=args.dry_run,
        )

    all_features_csv = Path(outputs_cfg.get("all_features_final_csv", out_root / "all_matches_features_final.csv"))
    all_model_suite_csv = Path(outputs_cfg.get("all_model_suite_csv", xg_out_root / "all_matches_model_suite.csv"))
    if not args.dry_run:
        all_df = pd.concat([pd.read_csv(p) for p in features_final_paths], ignore_index=True)
        all_features_csv.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(all_features_csv, index=False)
        print("saved:", all_features_csv)

    if bool(modeling.get("run_model_suite", False)):
        _run_step(
            [
                python_bin,
                str(project_root / "code" / "xg_baseline" / "classic_xg.py"),
                "--inputs",
                str(all_features_csv),
                "--run_model_suite",
                "--group_col",
                str(modeling.get("group_col", "game_id")),
                "--n_splits",
                str(modeling.get("n_splits", 3)),
                "--out_csv",
                str(all_model_suite_csv),
            ],
            dry_run=args.dry_run,
        )

    print("All-match pipeline completed.")


if __name__ == "__main__":
    main()

