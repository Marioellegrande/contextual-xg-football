"""
plot_shap_dependence.py
=======================
Figure D: SHAP dependence plots for the three most important features
in XGBoost M12, retrained on the full dataset.

SHAP values are computed on a model retrained on the full dataset
using the same specification as M12, to maximise stability of the
estimated feature effects (as stated in §8.6 of the thesis).
AUC/log-loss evaluation uses GroupKFold CV (separate from this script).

Three subplots:
  1. pressure_nd_dist_m  — dominant contextual feature
  2. angle_rad           — second-ranked by SHAP importance
  3. gk_lateral_offset   — leading non-pressure contextual feature

Usage:
    python code/visualization/plot_shap_dependence.py \
        --features code/data_pipeline/outputs/all_matches_features_final.csv \
        --out thesis/figures/fig_shap_dependence.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap", file=sys.stderr)
    sys.exit(1)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# M12 feature list (mirrors classic_xg.py)
# ---------------------------------------------------------------------------

M2_FEATS   = ["distance_m", "angle_rad", "goal_diff", "time_since_last_event_s"]
PRESSURE   = ["pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m"]
OBSTRUCT   = ["obstruction_count"]
GK         = ["gk_ball_distance", "gk_depth", "gk_lateral_offset"]
KINEM      = ["ball_speed_mps", "shooter_speed_mps"]
BODY       = ["shot_body_part"]
PATTERN    = ["play_pattern"]
STRUCT     = ["defender_dist_mean3", "possession_length", "fast_break"]

M12_FEATS  = M2_FEATS + PRESSURE + OBSTRUCT + GK + KINEM + BODY + PATTERN + STRUCT + \
             ["defender_dist_2", "free_angle"]

# Features to plot (in order of SHAP importance from §8.6)
PLOT_FEATURES = [
    ("pressure_nd_dist_m",  "Nearest-defender distance (m)",
     "Defensive pressure at moment of shot"),
    ("angle_rad",            "Shot angle (rad)",
     "Angular opening of goal from shooter"),
    ("gk_lateral_offset",   "Goalkeeper lateral offset (m)",
     "GK displacement from goal centre"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_features(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X = X.copy()
    cols = []
    for col in X.columns:
        if X[col].dtype == object or hasattr(X[col], "cat"):
            codes, _ = pd.factorize(X[col])
            X[col] = codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        cols.append(col)
    arr = X.to_numpy(dtype=float)
    # Impute with column medians
    medians = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        arr[np.isnan(arr[:, j]), j] = medians[j]
    return arr, cols


def train_m12_full(df: pd.DataFrame) -> tuple[XGBClassifier, np.ndarray, list[str]]:
    """Retrain XGBoost M12 on the full dataset."""
    available = [f for f in M12_FEATS if f in df.columns]
    missing   = [f for f in M12_FEATS if f not in df.columns]
    if missing:
        print(f"M12: missing columns (will be excluded): {missing}")

    y = df["is_goal"].astype(int).to_numpy()
    X_arr, feat_names = encode_features(df[available])

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    spw   = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_arr, y)
    print(f"M12 retrained on {len(df)} shots, {n_pos} goals, "
          f"{len(feat_names)} features")
    return model, X_arr, feat_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP dependence plots")
    parser.add_argument(
        "--features",
        default="code/data_pipeline/outputs/all_matches_features_final.csv",
        help="Path to combined features CSV",
    )
    parser.add_argument(
        "--out",
        default="thesis/figures/fig_shap_dependence.png",
        help="Output path for the figure",
    )
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: features file not found: {feat_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(feat_path)
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df.loc[df["is_goal"].isin([0, 1])].reset_index(drop=True)
    print(f"Loaded {len(df)} shots, {int(df['is_goal'].sum())} goals")

    # Retrain M12 on full dataset
    model, X_arr, feat_names = train_m12_full(df)

    # Compute SHAP values
    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)
    print(f"SHAP matrix shape: {shap_values.shape}")

    feat_idx = {name: i for i, name in enumerate(feat_names)}

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="white")
    fig.suptitle(
        "Figure D: SHAP dependence plots — XGBoost M12 (retrained on full dataset)",
        fontsize=11, fontweight="bold", y=1.02
    )

    colours = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, (feat, xlabel, subtitle), col in zip(axes, PLOT_FEATURES, colours):
        if feat not in feat_idx:
            ax.text(0.5, 0.5, f"Feature\n'{feat}'\nnot available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(subtitle)
            continue

        idx       = feat_idx[feat]
        feat_vals = X_arr[:, idx]
        shap_vals = shap_values[:, idx]
        is_goal   = df["is_goal"].astype(int).to_numpy()

        # Scatter: colour by outcome
        ax.scatter(
            feat_vals[is_goal == 0], shap_vals[is_goal == 0],
            c="#95a5a6", alpha=0.6, s=25, label="No goal", zorder=2
        )
        ax.scatter(
            feat_vals[is_goal == 1], shap_vals[is_goal == 1],
            c=col, alpha=0.9, s=55, label="Goal", zorder=3,
            edgecolors="black", linewidths=0.5
        )

        # Smoothed trend (LOWESS-style via rolling median)
        sort_idx  = np.argsort(feat_vals)
        x_sorted  = feat_vals[sort_idx]
        s_sorted  = shap_vals[sort_idx]
        window    = max(5, len(x_sorted) // 8)
        s_smooth  = pd.Series(s_sorted).rolling(window, center=True, min_periods=1).mean().values
        ax.plot(x_sorted, s_smooth, color=col, lw=2.5, zorder=4, label="Trend")

        # Zero line
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.6, zorder=1)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("SHAP value (log-odds contribution)", fontsize=8)
        ax.set_title(subtitle, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right" if feat != "pressure_nd_dist_m" else "lower right")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Print SHAP summary stats for each feature
    print("\nSHAP summary for plotted features:")
    for feat, xlabel, _ in PLOT_FEATURES:
        if feat not in feat_idx:
            continue
        idx = feat_idx[feat]
        sv  = shap_values[:, idx]
        fv  = X_arr[:, idx]
        print(f"  {feat:30s}  mean|SHAP|={np.abs(sv).mean():.4f}  "
              f"min_feat={fv.min():.2f}  max_feat={fv.max():.2f}  "
              f"SHAP_range=[{sv.min():.3f}, {sv.max():.3f}]")


if __name__ == "__main__":
    main()
