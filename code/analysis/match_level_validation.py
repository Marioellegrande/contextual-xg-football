"""
match_level_validation.py
=========================
Aggregates shot-level xG predictions to match level and evaluates
whether predicted team xG totals reflect actual goal counts.

Evaluation uses leave-one-match-out (LOMO) cross-validation:
  - Train on 3 matches, predict on the held-out 4th.
  - Sum per-team predicted probabilities → match xG.
  - Compare with actual goals via Pearson r, MAE, win prediction accuracy.

Models evaluated: M1 (geometry only), M3 (+ pressure), M9 (full contextual).

Case study match: SJE–SIF (2025-03-30, game_id 2515625)

Usage:
    python code/analysis/match_level_validation.py \
        --features code/data_pipeline/outputs/all_matches_features_final.csv \
        --out_csv code/analysis/outputs/match_level_xg.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("warning: xgboost not installed — XGBoost columns will be NaN")
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Feature sets (mirrors classic_xg.py)
# ---------------------------------------------------------------------------

M2_FEATS = ["distance_m", "angle_rad", "goal_diff", "time_since_last_event_s"]
PRESSURE  = ["pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m"]
OBSTRUCT  = ["obstruction_count"]
GK        = ["gk_ball_distance", "gk_depth", "gk_lateral_offset"]
KINEM     = ["ball_speed_mps", "shooter_speed_mps"]
BODY      = ["shot_body_part"]
PATTERN   = ["play_pattern"]
STRUCT    = ["defender_dist_mean3", "possession_length", "fast_break"]

MODEL_FEATURES = {
    "M1": ["distance_m", "angle_rad"],
    "M3": M2_FEATS + PRESSURE,
    "M9": M2_FEATS + PRESSURE + OBSTRUCT + GK + KINEM + BODY + PATTERN,
}

# Actual match scorelines from Opta XML (verified from srml matchresults files)
ACTUAL_SCORES = {
    2442545: {"home_team": "AGF", "away_team": "FCM",
              "home_goals": 1, "away_goals": 1,
              "home_team_id": 420, "away_team_id": 1000},
    2442546: {"home_team": "FCN", "away_team": "AAB",
              "home_goals": 3, "away_goals": 0,
              "home_team_id": 2592, "away_team_id": 401},
    2442547: {"home_team": "SIF", "away_team": "SJE",
              "home_goals": 1, "away_goals": 0,
              "home_team_id": 418, "away_team_id": 2827},
    2515625: {"home_team": "SJE", "away_team": "SIF",
              "home_goals": 2, "away_goals": 1,
              "home_team_id": 2827, "away_team_id": 418},
}

CASE_STUDY_GAME_ID = 2515625  # SJE–SIF (2025-03-30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_features(X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or hasattr(X[col], "cat"):
            codes, _ = pd.factorize(X[col])
            X[col] = codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.to_numpy(dtype=float)


def impute_median(X_tr: np.ndarray, X_te: np.ndarray):
    medians = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = medians[j]
        X_te[np.isnan(X_te[:, j]), j] = medians[j]
    return X_tr, X_te


def train_logreg(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr = LogisticRegression(
        solver="liblinear", max_iter=2000,
        class_weight="balanced", random_state=42
    )
    lr.fit(X_tr_s, y_tr)
    return lr.predict_proba(X_te_s)[:, 1]


def train_xgb(X_tr, y_tr, X_te):
    if not HAS_XGB:
        return np.full(len(X_te), np.nan)
    n_neg = int(np.sum(y_tr == 0))
    n_pos = int(np.sum(y_tr == 1))
    spw = n_neg / max(n_pos, 1)
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    xgb.fit(X_tr, y_tr)
    return xgb.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_match_level_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-match-out CV: train on 3 matches, predict on 4th.
    Returns shot-level DataFrame with predicted xG columns appended.
    """
    df = df.copy()
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df.loc[df["is_goal"].isin([0, 1])].reset_index(drop=True)
    y = df["is_goal"].astype(int).to_numpy()
    groups = df["game_id"].to_numpy()

    logo = LeaveOneGroupOut()

    # Initialise prediction columns
    for model_name in MODEL_FEATURES:
        df[f"xg_logreg_{model_name}"] = np.nan
        df[f"xg_xgb_{model_name}"]    = np.nan

    for train_idx, test_idx in logo.split(df, y, groups=groups):
        test_game = groups[test_idx[0]]
        print(f"  Fold: test match = {test_game} "
              f"({ACTUAL_SCORES[test_game]['home_team']}–"
              f"{ACTUAL_SCORES[test_game]['away_team']})")

        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            print(f"    skipping — single class in training fold")
            continue

        for model_name, feat_list in MODEL_FEATURES.items():
            available = [f for f in feat_list if f in df.columns]
            X_all = encode_features(df[available])
            X_tr = X_all[train_idx].copy()
            X_te = X_all[test_idx].copy()
            X_tr, X_te = impute_median(X_tr, X_te)

            p_lr  = train_logreg(X_tr, y_tr, X_te)
            p_xgb = train_xgb(X_tr, y_tr, X_te)

            df.loc[test_idx, f"xg_logreg_{model_name}"] = p_lr
            df.loc[test_idx, f"xg_xgb_{model_name}"]    = p_xgb

    return df


def aggregate_to_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum shot-level xG per team per match. Merge with actual scores.
    Returns one row per team per match.
    """
    rows = []
    for game_id, info in ACTUAL_SCORES.items():
        gdf = df[df["game_id"] == game_id]
        if gdf.empty:
            continue

        for side, team_id, actual_goals in [
            ("home", info["home_team_id"], info["home_goals"]),
            ("away", info["away_team_id"], info["away_goals"]),
        ]:
            team_name = info[f"{side}_team"]
            tdf = gdf[gdf["team_id"] == team_id]
            row = {
                "game_id":    game_id,
                "match":      f"{info['home_team']}–{info['away_team']}",
                "team":       team_name,
                "side":       side,
                "shots":      len(tdf),
                "actual_goals": actual_goals,
            }
            for model_name in MODEL_FEATURES:
                for model_type in ["logreg", "xgb"]:
                    col = f"xg_{model_type}_{model_name}"
                    if col in tdf.columns:
                        row[f"xg_{model_type}_{model_name}"] = tdf[col].sum()
            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(agg: pd.DataFrame, xg_col: str) -> dict:
    """Pearson r, MAE, win prediction accuracy for a given xG column."""
    valid = agg.dropna(subset=[xg_col, "actual_goals"])
    r, pval = pearsonr(valid[xg_col], valid["actual_goals"])
    mae = (valid[xg_col] - valid["actual_goals"]).abs().mean()

    # Win prediction: per match, does higher-xG team win?
    correct = 0
    total   = 0
    for game_id in valid["game_id"].unique():
        gdf = valid[valid["game_id"] == game_id]
        if len(gdf) != 2:
            continue
        home = gdf[gdf["side"] == "home"].iloc[0]
        away = gdf[gdf["side"] == "away"].iloc[0]
        # Predicted winner: team with higher xG
        pred_home_wins = home[xg_col] > away[xg_col]
        # Actual winner
        if home["actual_goals"] == away["actual_goals"]:
            continue  # skip draws
        actual_home_wins = home["actual_goals"] > away["actual_goals"]
        correct += int(pred_home_wins == actual_home_wins)
        total   += 1

    win_acc = correct / total if total > 0 else np.nan
    return {"pearson_r": round(r, 3), "p_value": round(pval, 3),
            "mae": round(mae, 3), "win_accuracy": round(win_acc, 3),
            "n_teams": len(valid), "n_decisive": total}


def print_case_study(df: pd.DataFrame, agg: pd.DataFrame):
    """Print case study for SJE–SIF (2515625)."""
    info = ACTUAL_SCORES[CASE_STUDY_GAME_ID]
    print(f"\n{'='*60}")
    print(f"CASE STUDY: {info['home_team']}–{info['away_team']} "
          f"(actual: {info['home_goals']}–{info['away_goals']})")
    print(f"{'='*60}")

    case_agg = agg[agg["game_id"] == CASE_STUDY_GAME_ID]
    for _, row in case_agg.iterrows():
        print(f"\n  {row['team']} ({row['side']}): "
              f"{int(row['shots'])} shots, {int(row['actual_goals'])} actual goals")
        for model_name in MODEL_FEATURES:
            for mt in ["logreg", "xgb"]:
                col = f"xg_{mt}_{model_name}"
                if col in row:
                    print(f"    {col}: {row[col]:.3f}")

    # Shot-level detail for case study
    case_shots = df[df["game_id"] == CASE_STUDY_GAME_ID].copy()
    goals = case_shots[case_shots["is_goal"] == 1]
    if not goals.empty:
        print(f"\n  Goal shots in this match ({len(goals)} goals):")
        for _, g in goals.iterrows():
            team_name = (info["home_team"] if g["team_id"] == info["home_team_id"]
                         else info["away_team"])
            nd = g.get("pressure_nd_dist_m", np.nan)
            xg_m3 = g.get("xg_xgb_M3", np.nan)
            print(f"    {team_name}: min={g['min']}, "
                  f"nd_dist={nd:.1f}m, xg_xgb_M3={xg_m3:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Match-level xG validation")
    parser.add_argument(
        "--features",
        default="code/data_pipeline/outputs/all_matches_features_final.csv",
        help="Path to combined features CSV",
    )
    parser.add_argument(
        "--out_csv",
        default="code/analysis/outputs/match_level_xg.csv",
        help="Output path for match-level aggregated xG",
    )
    args = parser.parse_args()

    # Load data
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: features file not found: {feat_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} shots from {df['game_id'].nunique()} matches")
    print(f"Goals: {df['is_goal'].sum()} / {len(df)} ({df['is_goal'].mean()*100:.1f}%)")

    # Run LOMO CV
    print("\nRunning leave-one-match-out validation...")
    df_pred = run_match_level_validation(df)

    # Aggregate to match level
    agg = aggregate_to_match(df_pred)

    # Save
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"\nMatch-level xG saved to: {out_path}")

    # Print match-level table
    print(f"\n{'='*60}")
    print("MATCH-LEVEL xG TABLE")
    print(f"{'='*60}")
    display_cols = ["match", "team", "actual_goals",
                    "xg_logreg_M1", "xg_xgb_M1",
                    "xg_logreg_M3", "xg_xgb_M3",
                    "xg_logreg_M9", "xg_xgb_M9"]
    display_cols = [c for c in display_cols if c in agg.columns]
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(agg[display_cols].to_string(index=False))

    # Metrics
    print(f"\n{'='*60}")
    print("METRICS BY MODEL")
    print(f"{'='*60}")
    for model_name in MODEL_FEATURES:
        for mt in ["logreg", "xgb"]:
            col = f"xg_{mt}_{model_name}"
            if col not in agg.columns:
                continue
            m = compute_metrics(agg, col)
            print(f"  {col:25s}  r={m['pearson_r']:+.3f}  "
                  f"MAE={m['mae']:.3f}  "
                  f"win_acc={m['win_accuracy']:.2f} "
                  f"({m['n_decisive']} decisive matches)")

    # Case study
    print_case_study(df_pred, agg)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
