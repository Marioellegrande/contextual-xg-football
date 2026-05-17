from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contextual xG thesis pipeline end-to-end."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to pipeline configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(
            f"Config file not found: {args.config}\n"
            f"Tip: pass --config <path> or run from the project root."
        )

    cfg = _load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    python_bin = sys.executable

    paths = cfg.get("paths", {})
    files = cfg.get("files", {})
    outputs = cfg.get("outputs", {})
    align = cfg.get("alignment", {})
    pressure = cfg.get("pressure", {})
    structure = cfg.get("structure", {})
    obstruction = cfg.get("obstruction", {})
    advanced_context = cfg.get("advanced_context", {})
    goalkeeper = cfg.get("goalkeeper", {})
    tempo = cfg.get("tempo", {})
    modeling = cfg.get("modeling", {})
    pipeline_cfg = cfg.get("pipeline", {})
    feature_toggles = pipeline_cfg.get("features", {})

    shots_clean_csv = Path(outputs["shots_clean_csv"])
    shots_aligned_csv = Path(outputs["shots_aligned_csv"])
    shots_pressure_csv = Path(outputs.get("shots_pressure_csv", shots_aligned_csv))
    shots_structure_csv = Path(outputs.get("shots_structure_csv", shots_pressure_csv))
    shots_advanced_context_csv = Path(outputs.get("shots_advanced_context_csv", shots_structure_csv))
    shots_obstruction_csv = Path(outputs.get("shots_obstruction_csv", shots_advanced_context_csv))
    shots_goalkeeper_csv = Path(outputs.get("shots_goalkeeper_csv", shots_obstruction_csv))
    shots_tempo_csv = Path(outputs.get("shots_tempo_csv", shots_goalkeeper_csv))
    features_final_csv = Path(outputs.get("features_final_csv", shots_tempo_csv))
    classic_xg_csv = Path(outputs["classic_xg_csv"])
    model_suite_csv = Path(outputs.get("model_suite_csv", classic_xg_csv.with_name("model_suite.csv")))

    # Step 1: Build cleaned shot table + geometry.
    _run_step(
        [
            python_bin,
            str(project_root / "code" / "data_pipeline" / "extract_shots_clean_from_eventdetails.py"),
            "--eventdetails_xml",
            str(files["event_f24_xml"]),
            "--metadata_xml",
            str(files["tracking_metadata_xml"]),
            "--out_csv",
            str(shots_clean_csv),
        ],
        dry_run=args.dry_run,
    )

    # Step 2: DataBallPy event-tracking synchronization.
    _run_step(
        [
            python_bin,
            str(project_root / "code" / "data_pipeline" / "event_tracking_alignment_databallpy.py"),
            "--shots_csv",
            str(shots_clean_csv),
            "--tracking_metadata_xml",
            str(files["tracking_metadata_xml"]),
            "--tracking_raw_file",
            str(files["tracking_raw_file"]),
            "--event_f7_xml",
            str(files["event_f7_xml"]),
            "--event_f24_xml",
            str(files["event_f24_xml"]),
            "--n_batches",
            str(align.get("n_batches", "smart")),
            "--offset",
            str(align.get("offset", 1.0)),
            "--out_csv",
            str(shots_aligned_csv),
        ],
        dry_run=args.dry_run,
    )

    # Steps 3–9: All contextual features in one pass (reads tracking JSONL once).
    # compute_all_features.py replaces the chain of individual feature scripts
    # (pressure, structure, obstruction, goalkeeper, tempo, advanced_context).
    _run_step(
        [
            python_bin,
            str(project_root / "code" / "features" / "compute_all_features.py"),
            "--shots_aligned_csv",
            str(shots_aligned_csv),
            "--tracking_jsonl",
            str(files["tracking_raw_file"]),
            "--out_csv",
            str(features_final_csv),
            "--pitch_length_m",
            str(pressure.get("pitch_length_m", 105.0)),
            "--pitch_width_m",
            str(pressure.get("pitch_width_m", 68.0)),
            "--goal_width_m",
            str(advanced_context.get("goal_width_m", 7.32)),
            "--d_front_mode",
            str(pressure.get("d_front_mode", "variable")),
            "--d_back",
            str(pressure.get("d_back", 3.0)),
            "--q",
            str(pressure.get("q", 1.75)),
            "--fast_break_threshold_s",
            str(advanced_context.get("fast_break_threshold_s", 8.0)),
        ],
        dry_run=args.dry_run,
    )

    # Step 10: Classic xG baseline.
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

    # Optional step 11: Run thesis model suite.
    if bool(modeling.get("run_model_suite", False)):
        _run_step(
            [
                python_bin,
                str(project_root / "code" / "xg_baseline" / "classic_xg.py"),
                "--inputs",
                str(features_final_csv),
                "--run_model_suite",
                "--group_col",
                str(modeling.get("group_col", "game_id")),
                "--n_splits",
                str(modeling.get("n_splits", 3)),
                "--out_csv",
                str(model_suite_csv),
            ],
            dry_run=args.dry_run,
        )

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
