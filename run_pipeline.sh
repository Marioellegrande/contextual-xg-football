#!/usr/bin/env bash
set -euo pipefail

# Single entry point for the thesis pipeline.
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --single-match
#   ./run_pipeline.sh --dry-run
#   ./run_pipeline.sh --all-matches
#   ./run_pipeline.sh --all-matches --dry-run

if [[ "${1:-}" == "" ]]; then
  python scripts/run_pipeline_all_matches.py --config config/config.yaml
elif [[ "${1:-}" == "--all-matches" ]]; then
  shift
  python scripts/run_pipeline_all_matches.py --config config/config.yaml "$@"
elif [[ "${1:-}" == "--single-match" ]]; then
  shift
  python scripts/run_pipeline.py --config config/config.yaml
elif [[ "${1:-}" == -* ]]; then
  python scripts/run_pipeline_all_matches.py --config config/config.yaml "$@"
else
  CONFIG_PATH="$1"
  shift
  python scripts/run_pipeline_all_matches.py --config "$CONFIG_PATH" "$@"
fi
