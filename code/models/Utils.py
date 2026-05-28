"""
Utils.py
========
Fælles hjælpefunktioner og feature-liste brugt på tværs af alle modeller.
Genskabt fra Utils.cpython-313.pyc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss

# ── Feature liste ─────────────────────────────────────────────────────────────

FEATURES = [
    "distance_m",
    "angle_rad",
    "goal_diff",
    "time_since_last_event_s",
    "shot_body_part",
    "play_pattern",
    "pressure_nd_dist_m",
    "pressure_def_count_r1m",
    "pressure_def_count_r2m",
    "obstruction_count",
    "gk_ball_distance",
    "gk_depth",
    "gk_lateral_offset",
    "ball_speed_mps",
    "shooter_speed_mps",
]

# ── Hjælpefunktioner ──────────────────────────────────────────────────────────

def load_shots(path: str | Path) -> pd.DataFrame:
    """Load og validér shots CSV."""
    df = pd.read_csv(path)
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df.loc[df["is_goal"].isin([0, 1])].reset_index(drop=True)
    return df


def _encode_features(df_sub: pd.DataFrame) -> np.ndarray:
    """Label-encod kategoriske kolonner og konvertér til numpy array."""
    X = df_sub.copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            codes, _ = pd.factorize(X[col])
            X[col] = codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.to_numpy(dtype=float)


def _impute_with_train_median(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Imputér NaN med trænings-median (ingen data leakage)."""
    medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = medians[j]
        X_test[np.isnan(X_test[:, j]), j]   = medians[j]
    return X_train, X_test


def _pooled_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Beregn AUC (roc_auc_score), rundet til 4 decimaler."""
    return round(float(roc_auc_score(y_true, y_prob)), 4)


def _pooled_ll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Beregn log-loss, rundet til 4 decimaler."""
    return round(float(log_loss(y_true, y_prob)), 4)
