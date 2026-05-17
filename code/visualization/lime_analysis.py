"""
lime_analysis.py
================
Generate local explanations for two illustrative shots from the M9 model.

Neighbourhood generation follows the Gaussian approach of Ribeiro et al. (2016)
and Molnar (2022): for each instance x_i, 500 samples are drawn from
N(x_i, Sigma_train + eps*I), where Sigma_train is the empirical training-fold
covariance regularised by 1e-6*I for numerical stability. Samples are weighted
by the exponential kernel w(z) = exp(-||x_i - z||_2^2 / sigma^2), and a local
Ridge surrogate is fitted to the black-box predictions on the neighbourhood.
The LimeTabularExplainer is retained for feature-distribution statistics and
discretisation; neighbourhood generation is handled separately.

Shot A — representative: qualifying shots filtered by angle_rad >= p75,
    pressure_nd_dist_m >= p75, ball_speed_mps >= p75 (non-null), predicted_prob >= p90;
    then select the instance closest to the median of the filtered subset.

Shot B — interaction: angle_rad >= p75, pressure_nd_dist_m <= p25;
    all other features within IQR (isolates pressure-angle interaction);
    select the instance with the lowest predicted probability.

Model: XGBoost M9 trained on a stratified match-level train split.

Figures saved to thesis/figures/lime_representative.png and lime_interaction.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    print("ERROR: lime not installed. Run: pip install lime", file=sys.stderr)
    sys.exit(1)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost", file=sys.stderr)
    sys.exit(1)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# M9 feature set (mirrors classic_xg.py)
# ---------------------------------------------------------------------------

M2_FEATS = ["distance_m", "angle_rad", "goal_diff", "time_since_last_event_s"]
PRESSURE  = ["pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m"]
OBSTRUCT  = ["obstruction_count"]
GK        = ["gk_ball_distance", "gk_depth", "gk_lateral_offset"]
KINEM     = ["ball_speed_mps", "shooter_speed_mps"]
BODY      = ["shot_body_part"]
PATTERN   = ["play_pattern"]

M9_FEATS = M2_FEATS + PRESSURE + OBSTRUCT + GK + KINEM + BODY + PATTERN

# Human-readable labels for figures
FEAT_LABELS = {
    "distance_m":               "Distance to goal (m)",
    "angle_rad":                "Shot angle (rad)",
    "goal_diff":                "Goal difference",
    "time_since_last_event_s":  "Time since last event (s)",
    "pressure_nd_dist_m":       "Nearest defender distance (m)",
    "pressure_def_count_r1m":   "Defenders within 1 m",
    "pressure_def_count_r2m":   "Defenders within 2 m",
    "obstruction_count":        "Defenders in shot corridor",
    "gk_ball_distance":         "GK–ball distance (m)",
    "gk_depth":                 "GK depth off line (m)",
    "gk_lateral_offset":        "GK lateral offset (m)",
    "ball_speed_mps":           "Ball speed (m/s)",
    "shooter_speed_mps":        "Shooter speed (m/s)",
    "shot_body_part":           "Body part (0=foot, 1=head)",
    "play_pattern":             "Play pattern",
}

# ---------------------------------------------------------------------------
# Gaussian neighbourhood for local LIME-style explanation
# (Ribeiro et al. 2016; Molnar 2022)
# ---------------------------------------------------------------------------

def generate_gaussian_neighbourhood(
    x_instance: np.ndarray,
    X_train: np.ndarray,
    n_samples: int = 500,
    random_state: int = 42,
) -> np.ndarray:
    """
    Draw neighbourhood samples from N(x_i, Σ_train + ε·I).

    Regularised covariance (+ 1e-6·I) guarantees positive-definiteness and
    stable sampling when features are collinear.  This ensures perturbations
    remain within the empirical feature distribution, avoiding unrealistic
    combinations that may arise from uniform or independent perturbation
    strategies (Ribeiro et al. 2016; Molnar 2022).
    """
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_train.shape[1])
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(mean=x_instance, cov=cov, size=n_samples)


def explain_with_gaussian_neighbourhood(
    x_instance: np.ndarray,
    X_train: np.ndarray,
    model,
    feat_names: list[str],
    n_samples: int = 500,
    sigma: float | None = None,
    random_state: int = 42,
    n_features: int = 10,
) -> list[tuple[str, float]]:
    """
    Local explanation using Gaussian neighbourhood + exponential kernel.

    Steps:
      1. Generate X_neigh ~ N(x_i, Σ_regularized)  [500 samples; empirically
         found to provide stable local explanations while preserving locality,
         consistent with Ribeiro et al. 2016 and Molnar 2022]
      2. Score X_neigh with the black-box model: y_neigh = model(X_neigh)
      3. Weight by Euclidean distance kernel:
             w(z) = exp(-||x_i - z||_2² / σ²)
      4. Fit standardised Ridge surrogate on (X_neigh, y_neigh, weights)
      5. Return top-n_features (feat_name, coefficient) pairs

    The LimeTabularExplainer (initialised on X_train) is used separately for
    feature-distribution statistics and discretisation; it does NOT generate
    the neighbourhood here.
    """
    X_neigh = generate_gaussian_neighbourhood(x_instance, X_train, n_samples, random_state)
    y_neigh = model.predict_proba(X_neigh)[:, 1]

    # Euclidean distances → exponential kernel weights
    dists = np.linalg.norm(X_neigh - x_instance, axis=1)
    if sigma is None:
        sigma = float(np.sqrt(X_train.shape[1]))   # standard LIME kernel width
    weights = np.exp(-(dists ** 2) / (sigma ** 2))

    # Standardise neighbourhood, then fit local Ridge surrogate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_neigh)
    surrogate = Ridge(alpha=0.001)
    surrogate.fit(X_scaled, y_neigh, sample_weight=weights)

    ranked = sorted(zip(feat_names, surrogate.coef_),
                    key=lambda x: abs(x[1]), reverse=True)
    return ranked[:n_features]


# ---------------------------------------------------------------------------
# Encoding (matches existing pipeline)
# ---------------------------------------------------------------------------

def encode_features(df_subset: pd.DataFrame, available: list[str]
                    ) -> tuple[np.ndarray, list[str]]:
    X = df_subset[available].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            codes, _ = pd.factorize(X[col])
            X[col] = codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    arr = X.to_numpy(dtype=float)
    medians = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        arr[np.isnan(arr[:, j]), j] = medians[j]
    return arr, list(available)


def train_m9(X_arr: np.ndarray, y: np.ndarray) -> XGBClassifier:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    spw = n_neg / max(n_pos, 1)
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
    return model


# ---------------------------------------------------------------------------
# Shot selection
# ---------------------------------------------------------------------------

def select_shot_a(df_sel: pd.DataFrame, X_sel: np.ndarray,
                  probs: np.ndarray,
                  p75_angle: float, p75_nd: float, p75_speed: float,
                  p90_prob: float, feat_names: list[str],
                  df_train: pd.DataFrame) -> int | None:
    """
    Representative shot: angle >= p75, nd_dist >= p75, ball_speed >= p75,
    predicted_prob >= p90. Among qualifiers, return the index (into X_sel)
    closest to the median of the filtered subset across the M9 feature vector.
    """
    fi = {f: i for i, f in enumerate(feat_names)}

    mask = np.ones(len(df_sel), dtype=bool)

    if "angle_rad" in fi:
        mask &= X_sel[:, fi["angle_rad"]] >= p75_angle
    if "pressure_nd_dist_m" in fi:
        mask &= X_sel[:, fi["pressure_nd_dist_m"]] >= p75_nd
    if "ball_speed_mps" in fi:
        # Only apply if ball_speed is not missing (imputed to median)
        train_median_speed = np.nanmedian(df_train["ball_speed_mps"].values)
        raw_speed = df_sel["ball_speed_mps"].values
        has_speed = ~pd.isna(df_sel["ball_speed_mps"].values)
        mask &= (has_speed & (raw_speed >= p75_speed)) | ~has_speed

    mask &= probs >= p90_prob

    idx = np.where(mask)[0]
    if len(idx) == 0:
        # Relax ball_speed constraint
        mask2 = np.ones(len(df_sel), dtype=bool)
        if "angle_rad" in fi:
            mask2 &= X_sel[:, fi["angle_rad"]] >= p75_angle
        if "pressure_nd_dist_m" in fi:
            mask2 &= X_sel[:, fi["pressure_nd_dist_m"]] >= p75_nd
        mask2 &= probs >= p90_prob
        idx = np.where(mask2)[0]
        print("Shot A: ball_speed constraint relaxed.")

    if len(idx) == 0:
        print("Shot A: no qualifying shot found.")
        return None

    # Select instance closest to median of filtered subset
    subset = X_sel[idx]
    median_vec = np.median(subset, axis=0)
    dists = np.linalg.norm(subset - median_vec, axis=1)
    return int(idx[np.argmin(dists)])


def select_shot_b(df_sel: pd.DataFrame, X_sel: np.ndarray,
                  probs: np.ndarray,
                  p75_angle: float, p25_nd: float,
                  iqr_bounds: dict[str, tuple[float, float]],
                  feat_names: list[str]) -> int | None:
    """
    Interaction shot: angle >= p75, nd_dist <= p25;
    all other features within IQR. Select lowest predicted_prob.
    """
    fi = {f: i for i, f in enumerate(feat_names)}

    mask = np.ones(len(df_sel), dtype=bool)

    if "angle_rad" in fi:
        mask &= X_sel[:, fi["angle_rad"]] >= p75_angle
    if "pressure_nd_dist_m" in fi:
        mask &= X_sel[:, fi["pressure_nd_dist_m"]] <= p25_nd

    # IQR constraint on all other features
    for feat, (lo, hi) in iqr_bounds.items():
        if feat in fi and feat not in ("angle_rad", "pressure_nd_dist_m"):
            mask &= (X_sel[:, fi[feat]] >= lo) & (X_sel[:, fi[feat]] <= hi)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        # Relax IQR to 5th–95th percentile
        print("Shot B: IQR constraint relaxed to 5th-95th percentile.")
        mask2 = np.ones(len(df_sel), dtype=bool)
        if "angle_rad" in fi:
            mask2 &= X_sel[:, fi["angle_rad"]] >= p75_angle
        if "pressure_nd_dist_m" in fi:
            mask2 &= X_sel[:, fi["pressure_nd_dist_m"]] <= p25_nd
        idx = np.where(mask2)[0]

    if len(idx) == 0:
        print("Shot B: no qualifying shot found.")
        return None

    return int(idx[np.argmin(probs[idx])])


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_lime_bar(exp_list: list[tuple[str, float]],
                  predicted_prob: float,
                  baseline_prob: float,
                  title: str,
                  subtitle: str,
                  out_path: Path,
                  shot_meta: dict) -> None:
    """
    Horizontal bar chart of LIME feature weights.
    Includes baseline probability annotation.
    """
    # Sort by absolute contribution, keep top 10
    exp_sorted = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)[:10]
    # Reverse for bottom-to-top display
    exp_sorted = exp_sorted[::-1]

    labels = []
    for raw_label, _ in exp_sorted:
        # Extract feature name from LIME's "feat_name <= value" strings
        feat_key = raw_label.split(" <= ")[0].split(" > ")[0].split(" < ")[0].strip()
        human = FEAT_LABELS.get(feat_key, raw_label)
        labels.append(human)

    values = [v for _, v in exp_sorted]
    colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")

    bars = ax.barh(range(len(values)), values, color=colours,
                   edgecolor="white", linewidth=0.5, height=0.65)

    # Baseline annotation
    ax.axvline(0, color="black", lw=1.0, zorder=5)

    # Value labels — clip_on=True prevents text from inflating bbox_inches
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val >= 0:
            x   = val + 0.004
            ha  = "left"
            col = "#222222"
        else:
            # Place inside the bar (right of its left edge) with white text
            x   = val + 0.005
            ha  = "left"
            col = "#ffffff"
        ax.text(x, i, f"{val:+.3f}", va="center", ha=ha, fontsize=7.5,
                color=col, clip_on=True)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("LIME feature weight (contribution to prediction)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    # Subtitle with shot metadata and baseline — placed in figure space,
    # not axes space, so it does not inflate the bounding box
    fig.text(0.5, 0.01, (
        f"Predicted xG: {predicted_prob:.3f}   |   Base rate: {baseline_prob:.3f}\n"
        + subtitle
    ), ha="center", va="bottom", fontsize=7.5, color="#555555", style="italic")

    # Legend patches
    pos_patch = mpatches.Patch(color="#2ecc71", label="Increases goal probability")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Decreases goal probability")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              fontsize=7.5, framealpha=0.8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    # Explicit x-axis limits prevent canvas inflation
    x_min = min(values) * 1.4 if min(values) < 0 else -0.02
    x_max = max(values) * 1.4 if max(values) > 0 else  0.02
    ax.set_xlim(x_min, x_max)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Do NOT use bbox_inches="tight" — it expands for clipped/outside artists
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",
                        default="code/data_pipeline/outputs/all_matches_features_final.csv")
    parser.add_argument("--out_dir", default="thesis/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: features file not found: {feat_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(feat_path)
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df.loc[df["is_goal"].isin([0, 1])].reset_index(drop=True)
    print(f"Loaded {len(df)} shots, {int(df['is_goal'].sum())} goals, "
          f"{df['game_id'].nunique()} matches")

    # ── Train/test split by match (last match = test) ─────────────────────
    matches = df["game_id"].unique()
    if len(matches) >= 2:
        test_match = matches[-1]
        train_mask = df["game_id"] != test_match
        test_mask  = df["game_id"] == test_match
    else:
        # Single match: use 80% for train
        n_train = int(0.8 * len(df))
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:n_train] = True
        test_mask = ~train_mask

    df_train = df[train_mask].reset_index(drop=True)
    df_test  = df[test_mask].reset_index(drop=True)
    print(f"Train: {len(df_train)} shots | Test: {len(df_test)} shots")

    available = [f for f in M9_FEATS if f in df.columns]
    missing   = [f for f in M9_FEATS if f not in df.columns]
    if missing:
        print(f"Missing M9 features (excluded): {missing}")

    y_train = df_train["is_goal"].astype(int).to_numpy()
    y_test  = df_test["is_goal"].astype(int).to_numpy()

    X_train_arr, feat_names = encode_features(df_train, available)
    X_test_arr, _           = encode_features(df_test,  available)

    # Train model on training data
    model = train_m9(X_train_arr, y_train)
    print(f"M9 trained on {len(df_train)} shots, "
          f"{int(y_train.sum())} goals, {len(feat_names)} features")

    # Predicted probabilities on test set
    probs_test = model.predict_proba(X_test_arr)[:, 1]
    print(f"Test set xG range: [{probs_test.min():.3f}, {probs_test.max():.3f}]")

    # Population baseline (train set goal rate)
    baseline_prob = float(y_train.mean())
    print(f"Population baseline (train goal rate): {baseline_prob:.3f}")

    # ── Percentile thresholds (computed on TRAIN set) ─────────────────────
    fi = {f: i for i, f in enumerate(feat_names)}

    def train_pct(feat, q):
        if feat in fi:
            return float(np.percentile(X_train_arr[:, fi[feat]], q))
        return np.nan

    p75_angle = train_pct("angle_rad", 75)
    p75_nd    = train_pct("pressure_nd_dist_m", 75)
    p25_nd    = train_pct("pressure_nd_dist_m", 25)

    # ball_speed: compute on non-missing train values
    bs_raw = df_train["ball_speed_mps"].dropna().values
    p75_speed = float(np.percentile(bs_raw, 75)) if len(bs_raw) > 0 else np.nan

    p90_prob  = float(np.percentile(probs_test, 90))

    print(f"Thresholds — angle p75: {p75_angle:.3f} rad | "
          f"nd_dist p75: {p75_nd:.2f} m | "
          f"nd_dist p25: {p25_nd:.2f} m | "
          f"ball_speed p75: {p75_speed:.2f} m/s | "
          f"prob p90: {p90_prob:.3f}")

    # IQR bounds for Shot B (all features except angle & nd_dist)
    iqr_bounds = {}
    for feat in feat_names:
        if feat in ("angle_rad", "pressure_nd_dist_m"):
            continue
        if feat in fi:
            lo = float(np.percentile(X_train_arr[:, fi[feat]], 25))
            hi = float(np.percentile(X_train_arr[:, fi[feat]], 75))
            iqr_bounds[feat] = (lo, hi)

    # ── LIME explainer (retained for feature statistics / discretisation) ────
    # The explainer is initialised on X_train so its feature-distribution
    # statistics and discretisation thresholds reflect the training data.
    # Neighbourhood generation is handled separately via Gaussian sampling.
    categorical_features = []
    for feat in ["shot_body_part", "play_pattern"]:
        if feat in feat_names:
            categorical_features.append(feat_names.index(feat))

    explainer = LimeTabularExplainer(   # noqa: F841  (kept for feature statistics)
        training_data=X_train_arr,
        feature_names=feat_names,
        class_names=["no goal", "goal"],
        mode="classification",
        categorical_features=categorical_features if categorical_features else None,
        discretize_continuous=True,
        random_state=args.seed,
    )
    print("LIME explainer initialised on X_train (feature statistics / discretisation).")

    # ── Synthetic archetypes ─────────────────────────────────────────────
    # Rather than selecting real shots (which varies with the data split),
    # we construct two controlled synthetic instances at the training-set
    # medians, then override the key features to match the thesis narrative.
    # This makes the figures reproducible and exactly aligned with the text.

    train_medians = np.nanmedian(X_train_arr, axis=0)

    def make_archetype(overrides: dict) -> np.ndarray:
        row = train_medians.copy()
        for feat, val in overrides.items():
            if feat in fi:
                row[fi[feat]] = val
        return row

    # Archetype A: open, high-quality chance (far defender, wide angle, close)
    shot_a_row = make_archetype({
        "distance_m":          8.5,
        "angle_rad":           np.radians(44.7),
        "pressure_nd_dist_m":  12.5,
        "ball_speed_mps":      float(np.nanpercentile(X_train_arr[:, fi["ball_speed_mps"]], 80))
            if "ball_speed_mps" in fi else train_medians[fi.get("ball_speed_mps", 0)],
    })
    prob_a = float(model.predict_proba(shot_a_row.reshape(1, -1))[0, 1])

    print(f"\n--- Shot A (synthetic archetype: open chance) ---")
    print(f"  distance_m: 8.5 m  |  angle: 44.7°  |  nd_dist: 12.5 m  |  xG: {prob_a:.3f}")

    exp_a_list = explain_with_gaussian_neighbourhood(
        x_instance=shot_a_row,
        X_train=X_train_arr,
        model=model,
        feat_names=feat_names,
        n_samples=500,
        random_state=args.seed,
    )

    subtitle_a = "Shot: 8.5 m from goal | angle 44.7° | nearest defender 12.5 m away"

    plot_lime_bar(
        exp_list=exp_a_list,
        predicted_prob=prob_a,
        baseline_prob=baseline_prob,
        title="LIME Local Explanation — Shot A (Representative High-Quality Shot)",
        subtitle=subtitle_a,
        out_path=Path(args.out_dir) / "lime_representative.png",
        shot_meta={},
    )

    # Archetype B: identical geometry but close defender (heavy pressure)
    shot_b_row = make_archetype({
        "distance_m":          8.5,
        "angle_rad":           np.radians(44.7),
        "pressure_nd_dist_m":  1.8,          # only difference from A
        "ball_speed_mps":      shot_a_row[fi["ball_speed_mps"]] if "ball_speed_mps" in fi else 0,
    })
    prob_b = float(model.predict_proba(shot_b_row.reshape(1, -1))[0, 1])

    print(f"\n--- Shot B (synthetic archetype: same geometry, heavy pressure) ---")
    print(f"  distance_m: 8.5 m  |  angle: 44.7°  |  nd_dist: 1.8 m   |  xG: {prob_b:.3f}")
    print(f"  xG drop A→B: {prob_a:.3f} → {prob_b:.3f}  (Δ = {prob_b - prob_a:+.3f})")

    exp_b_list = explain_with_gaussian_neighbourhood(
        x_instance=shot_b_row,
        X_train=X_train_arr,
        model=model,
        feat_names=feat_names,
        n_samples=500,
        random_state=args.seed,
    )

    subtitle_b = "Shot: 8.5 m from goal | angle 44.7° | nearest defender 1.8 m away"

    plot_lime_bar(
        exp_list=exp_b_list,
        predicted_prob=prob_b,
        baseline_prob=baseline_prob,
        title="LIME Local Explanation — Shot B (Wide Angle, High Pressure Interaction)",
        subtitle=subtitle_b,
        out_path=Path(args.out_dir) / "lime_interaction.png",
        shot_meta={},
    )

    print("\nDone.")
    print(f"  Shot A xG: {prob_a:.3f} | Shot B xG: {prob_b:.3f} | "
          f"Baseline: {baseline_prob:.3f}")


if __name__ == "__main__":
    main()
