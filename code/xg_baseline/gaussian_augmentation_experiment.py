"""
gaussian_augmentation_experiment.py
=====================================
Supplementary robustness experiment: Gaussian data augmentation.

PURPOSE
-------
This script is NOT part of the main modelling pipeline.
It serves as a robustness check only and does not contribute to the primary
model comparison or conclusions (Section 8.3 / Table tab:model-comparison).

WHAT IT DOES
------------
For each GroupKFold training fold:
  1. Fit N(mu, Sigma_regularised) on the real training data.
  2. Generate N_AUG synthetic observations from that distribution.
  3. Assign pseudo-labels using the baseline model (trained on real data only).
  4. Retrain on [real + synthetic] and evaluate on the real held-out fold.
  5. Compare AUC/log-loss vs. baseline (real data only).

IMPORTANT CAVEATS (stated explicitly in thesis Appendix B)
----------------------------------------------------------
- Pseudo-labels are model predictions, NOT ground truth.
- Synthetic samples are therefore not independent observations, introducing
  a form of self-training bias.
- Any performance difference (positive or negative) is exploratory only.
- Gaussian sampling here is ENTIRELY SEPARATE from the Gaussian neighbourhood
  sampling used in LIME (lime_analysis.py). That procedure generates local
  perturbations for explanation; it never enters model training.

REFS: James et al. (2013); Chawla et al. (2002); Ribeiro et al. (2016)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants — must match XGB_CONFIG in classic_xg.py
# ---------------------------------------------------------------------------
N_CV_SPLITS: int = 5
N_AUG: int = 2000       # synthetic samples per fold
EPS: float = 1e-6       # covariance regularisation (positive-definiteness)
RANDOM_STATE: int = 42

# Base goal rate (used as pseudo-label threshold) — computed from data
BASE_RATE: float | None = None   # set from data if None

# M9 feature set — same as main pipeline
M9_FEATS = [
    "distance_m", "angle_rad", "goal_diff", "time_since_last_event_s",
    "pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m",
    "obstruction_count",
    "gk_ball_distance", "gk_depth", "gk_lateral_offset",
    "ball_speed_mps", "shooter_speed_mps",
    "shot_body_part", "play_pattern",
]

XGB_CONFIG = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_and_impute(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    arr = X.to_numpy(dtype=float)
    col_medians = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        arr[np.isnan(arr[:, j]), j] = col_medians[j]
    return arr


def generate_gaussian_samples(
    X_train: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw synthetic observations from N(mu, Sigma + eps*I).
    Regularised covariance guarantees positive-definiteness.
    """
    mu = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + EPS * np.eye(X_train.shape[1])
    return rng.multivariate_normal(mu, cov, size=n_samples)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_augmentation_experiment(
    features_csv: str,
    out_csv: str,
) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df[df["is_goal"].isin([0, 1])].reset_index(drop=True)

    available = [f for f in M9_FEATS if f in df.columns]
    missing = [f for f in M9_FEATS if f not in df.columns]
    if missing:
        print(f"WARNING: missing M9 features (excluded): {missing}")

    X_all = encode_and_impute(df, available)
    y_all = df["is_goal"].astype(int).to_numpy()
    groups = df["game_id"].to_numpy()

    base_rate = float(y_all.mean())
    print(f"Dataset: {len(df)} shots, {int(y_all.sum())} goals, "
          f"base rate = {base_rate:.4f}, {df['game_id'].nunique()} matches")

    gkf = GroupKFold(n_splits=N_CV_SPLITS)
    rng = np.random.default_rng(RANDOM_STATE)
    results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_all, y_all, groups)
    ):
        print(f"\n--- Fold {fold + 1}/{N_CV_SPLITS} ---")
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        spw = n_neg / max(n_pos, 1)

        # --- baseline: real data only ---
        model_base = XGBClassifier(**XGB_CONFIG, scale_pos_weight=spw)
        model_base.fit(X_tr, y_tr)
        p_base = model_base.predict_proba(X_te)[:, 1]
        auc_base = roc_auc_score(y_te, p_base)
        ll_base = log_loss(y_te, p_base)
        print(f"  Baseline   → AUC={auc_base:.4f}  LL={ll_base:.4f}")

        # --- Gaussian augmentation ---
        # NOTE: pseudo-labels come from baseline model (self-training bias)
        X_aug = generate_gaussian_samples(X_tr, N_AUG, rng)
        p_aug_prob = model_base.predict_proba(X_aug)[:, 1]
        y_aug = (p_aug_prob >= base_rate).astype(int)   # threshold = base rate

        X_combined = np.vstack([X_tr, X_aug])
        y_combined = np.concatenate([y_tr, y_aug])

        n_neg_c = int((y_combined == 0).sum())
        n_pos_c = int((y_combined == 1).sum())
        spw_c = n_neg_c / max(n_pos_c, 1)

        model_aug = XGBClassifier(**XGB_CONFIG, scale_pos_weight=spw_c)
        model_aug.fit(X_combined, y_combined)
        p_aug_out = model_aug.predict_proba(X_te)[:, 1]
        auc_aug = roc_auc_score(y_te, p_aug_out)
        ll_aug = log_loss(y_te, p_aug_out)
        print(f"  Augmented  → AUC={auc_aug:.4f}  LL={ll_aug:.4f}  "
              f"(ΔAUC={auc_aug - auc_base:+.4f})")

        results.append({
            "fold": fold + 1,
            "n_train_real": len(X_tr),
            "n_synthetic": N_AUG,
            "auc_baseline": round(auc_base, 4),
            "logloss_baseline": round(ll_base, 4),
            "auc_augmented": round(auc_aug, 4),
            "logloss_augmented": round(ll_aug, 4),
            "delta_auc": round(auc_aug - auc_base, 4),
            "delta_logloss": round(ll_aug - ll_base, 4),
        })

    results_df = pd.DataFrame(results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY (mean across folds) ===")
    print(results_df[["auc_baseline", "auc_augmented", "delta_auc",
                       "logloss_baseline", "logloss_augmented",
                       "delta_logloss"]].mean().round(4).to_string())
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/all_matches_features_final.csv",
    )
    parser.add_argument(
        "--out_csv",
        default="outputs/gaussian_augmentation_results.csv",
    )
    args = parser.parse_args()

    run_augmentation_experiment(args.features, args.out_csv)
