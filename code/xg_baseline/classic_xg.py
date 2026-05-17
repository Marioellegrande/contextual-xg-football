from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_COLS = ["is_goal", "distance_m", "angle_rad"]
OPEN_PLAY_TYPES = {"regular_play", "corner_kick", "crossed_free_kick", "counter_attack"}


# ---------------------------------------------------------------------------
# M1–M13 thesis model suite definition
# Feature names match the actual columns produced by the feature pipeline.
# ---------------------------------------------------------------------------

def get_thesis_model_suite() -> Dict[str, List[str]]:
    """Return ordered dict of model name → feature list for M1–M13."""
    m2 = ["distance_m", "angle_rad", "goal_diff", "time_since_last_event_s"]
    pressure = ["pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m"]
    obstruction = ["obstruction_count"]
    goalkeeper = ["gk_ball_distance", "gk_depth", "gk_lateral_offset"]
    kinematics = ["ball_speed_mps", "shooter_speed_mps"]
    body_part = ["shot_body_part"]
    play_pat = ["play_pattern"]
    m9_all = m2 + pressure + obstruction + goalkeeper + kinematics + body_part + play_pat
    structure = ["defender_dist_mean3", "possession_length", "fast_break"]
    # def_in_r5m disambiguates "no defender within 5m" (NaN speed, indicator=0)
    # from "slow defender present" (speed measured, indicator=1).
    movement = ["def_speed_mean_r5m", "closing_speed_mean_r5m", "def_in_r5m"]
    m12_all = m9_all + structure + ["defender_dist_2", "free_angle"]

    return {
        "M1": ["distance_m", "angle_rad"],
        "M2": m2,
        "M3": m2 + pressure,
        "M4": m2 + obstruction,
        "M5": m2 + goalkeeper,
        "M6": m2 + kinematics,
        "M7": m2 + body_part,
        "M8": m2 + play_pat,
        "M9": m9_all,
        "M10": m9_all + structure,
        "M11": m9_all + structure + ["defender_dist_2"],
        "M12": m12_all,
        # M13: full contextual model — M12 + defender movement features.
        # This is the complete feature set evaluated in the thesis.
        "M13": m12_all + movement,
    }


# ---------------------------------------------------------------------------
# Shared constants — single source of truth for all CV and model config
# ---------------------------------------------------------------------------

N_CV_SPLITS: int = 5  # GroupKFold splits; must match thesis text (Chapter 6)

# Fixed XGBoost hyperparameters applied identically across all experiments
# (main suite, ablation, greedy selection). Do not override per-function.
XGB_CONFIG: dict = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)


def _encode_features(X: pd.DataFrame) -> np.ndarray:
    """Convert a feature DataFrame to a float array, encoding categoricals."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or hasattr(X[col], "cat"):
            codes, _ = pd.factorize(X[col])
            X[col] = codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.to_numpy(dtype=float)


def _impute_with_train_median(
    X_tr: np.ndarray, X_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaN in both arrays using column medians from the training set.
    Falls back to 0 for columns that are entirely NaN in the training fold."""
    medians = np.nanmedian(X_tr, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)  # all-NaN column → 0
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = medians[j]
        X_te[np.isnan(X_te[:, j]), j] = medians[j]
    return X_tr, X_te


def run_model_suite(
    df: pd.DataFrame,
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
    run_mlp: bool = False,
) -> pd.DataFrame:
    """
    Run M1–M13 model suite with GroupKFold CV.

    Returns a DataFrame with columns:
      model, features, n_rows, logreg_ll, logreg_auc, xgb_ll, xgb_auc,
      tabpfn_ll, tabpfn_auc[, mlp_ll, mlp_auc]
    """
    import os as _os
    _tabpfn_token = _os.environ.get("TABPFN_TOKEN", "")
    print(f"[DEBUG] TABPFN_TOKEN present: {bool(_tabpfn_token)}, length: {len(_tabpfn_token)}")
    if _tabpfn_token:
        _os.environ["TABPFN_TOKEN"] = _tabpfn_token
        _os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    try:
        from tabpfn import TabPFNClassifier  # type: ignore
        has_tabpfn = True
        print("note: TabPFN imported successfully")
    except ImportError as e:
        print(f"note: tabpfn not installed — TabPFN columns will be NaN ({e})")
        has_tabpfn = False
    except Exception as e:
        print(f"note: TabPFN import failed — TabPFN columns will be NaN ({e})")
        has_tabpfn = False

    try:
        from xgboost import XGBClassifier  # type: ignore
        has_xgb = True
    except ImportError:
        print("note: xgboost not installed — XGBoost columns will be NaN")
        has_xgb = False

    if run_mlp:
        from sklearn.neural_network import MLPClassifier
        from sklearn.utils import compute_sample_weight

    model_suite = get_thesis_model_suite()

    df = df.copy()
    df["is_goal"] = pd.to_numeric(df["is_goal"], errors="coerce")
    df = df.loc[df["is_goal"].isin([0, 1])].reset_index(drop=True)
    y = df["is_goal"].astype(int).to_numpy()

    if group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        print(f"warning: group_col '{group_col}' not found — using sequential groups")
        groups = np.arange(len(df))

    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < n_splits:
        print(f"note: only {n_groups} groups available — using n_splits={actual_splits}")
    cv = GroupKFold(n_splits=actual_splits)

    results = []
    for model_name, feature_list in model_suite.items():
        available = [f for f in feature_list if f in df.columns]
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            print(f"{model_name}: missing columns (will be skipped): {missing}")

        X_full = _encode_features(df[available])

        # Pooled predictions: accumulate test-fold outputs, compute metrics once.
        pool_y_lr: List[int] = []
        pool_p_lr: List[float] = []
        pool_y_xgb: List[int] = []
        pool_p_xgb: List[float] = []
        pool_y_tab: List[int] = []
        pool_p_tab: List[float] = []
        pool_y_mlp: List[int] = []
        pool_p_mlp: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X_full, y, groups=groups)
        ):
            X_tr = X_full[train_idx].copy()
            X_te = X_full[test_idx].copy()
            y_tr = y[train_idx]
            y_te = y[test_idx]

            # Need both classes in training to fit the model.
            if len(np.unique(y_tr)) < 2:
                print(f"  {model_name} fold {fold_idx}: skipping (single class in train)")
                continue

            X_tr, X_te = _impute_with_train_median(X_tr, X_te)

            # --- Logistic Regression ---
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            lr = LogisticRegression(
                solver="liblinear", max_iter=2000,
                class_weight="balanced",   # handles ~7.6% positive rate
                random_state=random_state,
            )
            lr.fit(X_tr_s, y_tr)
            p_lr = lr.predict_proba(X_te_s)[:, 1]
            pool_y_lr.extend(y_te.tolist())
            pool_p_lr.extend(p_lr.tolist())

            # --- XGBoost ---
            if has_xgb:
                try:
                    _n_neg = int(np.sum(y_tr == 0))
                    _n_pos = int(np.sum(y_tr == 1))
                    _spw = _n_neg / max(_n_pos, 1)
                    xgb = XGBClassifier(
                        **XGB_CONFIG,
                        scale_pos_weight=_spw,
                    )
                    xgb.fit(X_tr, y_tr)
                    p_xgb = xgb.predict_proba(X_te)[:, 1]
                    pool_y_xgb.extend(y_te.tolist())
                    pool_p_xgb.extend(p_xgb.tolist())
                except Exception as exc:
                    print(f"  {model_name} fold {fold_idx}: XGBoost error — {exc}")

            # --- TabPFN ---
            if has_tabpfn:
                try:
                    tabpfn = TabPFNClassifier(device="cpu")
                    tabpfn.fit(X_tr, y_tr)
                    p_tab = tabpfn.predict_proba(X_te)[:, 1]
                    pool_y_tab.extend(y_te.tolist())
                    pool_p_tab.extend(p_tab.tolist())
                except Exception as exc:
                    print(f"  {model_name} fold {fold_idx}: TabPFN error — {type(exc).__name__}: {exc}")

            # --- MLP --- (MLPClassifier does not support sample_weight)
            if run_mlp:
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=random_state,
                    )
                    mlp.fit(X_tr_s, y_tr)
                    p_mlp = mlp.predict_proba(X_te_s)[:, 1]
                    pool_y_mlp.extend(y_te.tolist())
                    pool_p_mlp.extend(p_mlp.tolist())
                except Exception as exc:
                    print(f"  {model_name} fold {fold_idx}: MLP error — {exc}")

        row: Dict = {
            "model": model_name,
            "features": ",".join(available),
            "n_rows": len(df),
            "logreg_ll": _pooled_ll(pool_y_lr, pool_p_lr),
            "logreg_auc": _pooled_auc(pool_y_lr, pool_p_lr),
            "xgb_ll": _pooled_ll(pool_y_xgb, pool_p_xgb) if has_xgb else None,
            "xgb_auc": _pooled_auc(pool_y_xgb, pool_p_xgb) if has_xgb else None,
            "tabpfn_ll": _pooled_ll(pool_y_tab, pool_p_tab) if has_tabpfn else None,
            "tabpfn_auc": _pooled_auc(pool_y_tab, pool_p_tab) if has_tabpfn else None,
            "mlp_ll": _pooled_ll(pool_y_mlp, pool_p_mlp) if run_mlp else None,
            "mlp_auc": _pooled_auc(pool_y_mlp, pool_p_mlp) if run_mlp else None,
        }
        results.append(row)

        lr_str = (
            f"ll={row['logreg_ll']:.4f}, AUC={row['logreg_auc']:.4f}"
            if row["logreg_ll"] is not None
            else "no valid folds"
        )
        xgb_str = ""
        if has_xgb and row["xgb_ll"] is not None:
            xgb_str = f" | XGBoost ll={row['xgb_ll']:.4f}, AUC={row['xgb_auc']:.4f}"
        tab_str = ""
        if has_tabpfn and row["tabpfn_ll"] is not None:
            tab_str = f" | TabPFN ll={row['tabpfn_ll']:.4f}, AUC={row['tabpfn_auc']:.4f}"
        mlp_str = ""
        if run_mlp and row["mlp_ll"] is not None:
            mlp_str = f" | MLP ll={row['mlp_ll']:.4f}, AUC={row['mlp_auc']:.4f}"
        print(f"{model_name}: LogReg {lr_str}{xgb_str}{tab_str}{mlp_str}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Module-level pooled metric helpers (used by both run_model_suite and ablation)
# ---------------------------------------------------------------------------

def _pooled_auc(ys: List[int], ps: List[float]) -> Optional[float]:
    if not ys or len(np.unique(ys)) < 2:
        return None
    return round(float(roc_auc_score(ys, ps)), 4)


def _pooled_ll(ys: List[int], ps: List[float]) -> Optional[float]:
    if not ys or len(np.unique(ys)) < 2:
        return None
    return round(float(log_loss(ys, ps)), 4)


# ---------------------------------------------------------------------------
# Backward feature ablation
# ---------------------------------------------------------------------------

ABLATION_GROUPS: Dict[str, List[str]] = {
    "pressure":    ["pressure_nd_dist_m", "pressure_def_count_r1m", "pressure_def_count_r2m"],
    "goalkeeper":  ["gk_ball_distance", "gk_depth", "gk_lateral_offset"],
    "obstruction": ["obstruction_count"],
}


def run_backward_ablation(
    df: pd.DataFrame,
    base_model: str = "M9",
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Backward feature ablation: remove one group at a time from base_model.

    Returns a DataFrame with columns:
        variant, removed_group, logreg_ll, logreg_auc, xgb_ll, xgb_auc,
        tabpfn_ll, tabpfn_auc, mlp_ll, mlp_auc
    """
    try:
        from xgboost import XGBClassifier  # type: ignore
        has_xgb = True
    except ImportError:
        has_xgb = False

    try:
        from tabpfn import TabPFNClassifier  # type: ignore
        has_tabpfn = True
    except ImportError:
        has_tabpfn = False

    from sklearn.neural_network import MLPClassifier

    suite = get_thesis_model_suite()
    base_features = suite[base_model]

    groups = df[group_col].values
    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    cv = GroupKFold(n_splits=actual_splits)

    def _run_variant(feature_list: List[str], label: str, removed: str) -> Dict:
        available = [f for f in feature_list if f in df.columns]
        X = _encode_features(df[available])
        y = df["is_goal"].values.astype(int)

        pool_y_lr: List[int] = []
        pool_p_lr: List[float] = []
        pool_y_xgb: List[int] = []
        pool_p_xgb: List[float] = []
        pool_y_tab: List[int] = []
        pool_p_tab: List[float] = []
        pool_y_mlp: List[int] = []
        pool_p_mlp: List[float] = []

        for tr_idx, te_idx in cv.split(X, y, groups=groups):
            X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
            y_tr, y_te = y[tr_idx], y[te_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            X_tr, X_te = _impute_with_train_median(X_tr, X_te)

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            lr = LogisticRegression(
                solver="liblinear", max_iter=2000,
                class_weight="balanced", random_state=random_state
            )
            lr.fit(X_tr_s, y_tr)
            pool_y_lr.extend(y_te.tolist())
            pool_p_lr.extend(lr.predict_proba(X_te_s)[:, 1].tolist())

            if has_xgb:
                _n_neg = int(np.sum(y_tr == 0))
                _n_pos = int(np.sum(y_tr == 1))
                _spw = _n_neg / max(_n_pos, 1)
                xgb = XGBClassifier(
                    **XGB_CONFIG,
                    scale_pos_weight=_spw,
                )
                xgb.fit(X_tr, y_tr)
                pool_y_xgb.extend(y_te.tolist())
                pool_p_xgb.extend(xgb.predict_proba(X_te)[:, 1].tolist())

            if has_tabpfn:
                try:
                    tabpfn = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                    tabpfn.fit(X_tr, y_tr)  # raw, unscaled — consistent with run_model_suite
                    pool_y_tab.extend(y_te.tolist())
                    pool_p_tab.extend(tabpfn.predict_proba(X_te)[:, 1].tolist())
                except Exception:
                    pass

            # MLP — MLPClassifier does not support sample_weight; use class_weight via balanced
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
            try:
                mlp.fit(X_tr_s, y_tr)
                pool_y_mlp.extend(y_te.tolist())
                pool_p_mlp.extend(mlp.predict_proba(X_te_s)[:, 1].tolist())
            except Exception:
                pass

        return {
            "variant": label,
            "removed_group": removed,
            "logreg_ll": _pooled_ll(pool_y_lr, pool_p_lr),
            "logreg_auc": _pooled_auc(pool_y_lr, pool_p_lr),
            "xgb_ll": _pooled_ll(pool_y_xgb, pool_p_xgb) if has_xgb else None,
            "xgb_auc": _pooled_auc(pool_y_xgb, pool_p_xgb) if has_xgb else None,
            "tabpfn_ll": _pooled_ll(pool_y_tab, pool_p_tab) if has_tabpfn else None,
            "tabpfn_auc": _pooled_auc(pool_y_tab, pool_p_tab) if has_tabpfn else None,
            "mlp_ll": _pooled_ll(pool_y_mlp, pool_p_mlp),
            "mlp_auc": _pooled_auc(pool_y_mlp, pool_p_mlp),
        }

    rows = []
    # Full model baseline
    rows.append(_run_variant(base_features, f"{base_model} (full)", "none"))
    # Remove each group
    for grp_name, grp_feats in ABLATION_GROUPS.items():
        reduced = [f for f in base_features if f not in grp_feats]
        rows.append(_run_variant(reduced, f"{base_model} \u2212{grp_name}", grp_name))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared CV helper for data-driven selection experiments
# ---------------------------------------------------------------------------

def _eval_features_xgb_cv(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Evaluate an XGBoost model with GroupKFold CV on the given feature list.

    Returns (pooled_log_loss, pooled_auc) or (None, None) if insufficient labels.
    Uses the same hyperparameters as run_model_suite().
    """
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        raise ImportError("xgboost is required for data-driven feature selection experiments.")

    available = [f for f in features if f in df.columns]
    if not available:
        return None, None

    y = df["is_goal"].astype(int).to_numpy()
    X_full = _encode_features(df[available])

    if group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        groups = np.arange(len(df))

    actual_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=actual_splits)

    pool_y: List[int] = []
    pool_p: List[float] = []

    for train_idx, test_idx in cv.split(X_full, y, groups=groups):
        X_tr, X_te = X_full[train_idx].copy(), X_full[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2:
            continue

        X_tr, X_te = _impute_with_train_median(X_tr, X_te)

        _n_neg = int(np.sum(y_tr == 0))
        _n_pos = int(np.sum(y_tr == 1))
        _spw = _n_neg / max(_n_pos, 1)

        xgb = XGBClassifier(
            **XGB_CONFIG,
            scale_pos_weight=_spw,
        )
        xgb.fit(X_tr, y_tr)
        p = xgb.predict_proba(X_te)[:, 1]
        pool_y.extend(y_te.tolist())
        pool_p.extend(p.tolist())

    return _pooled_ll(pool_y, pool_p), _pooled_auc(pool_y, pool_p)


# ---------------------------------------------------------------------------
# Greedy forward feature selection
# ---------------------------------------------------------------------------

def run_greedy_forward_selection(
    df: pd.DataFrame,
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Greedy forward feature selection driven by XGBoost GroupKFold CV log-loss.

    Starts from an empty feature set and iteratively adds the feature from the
    full M13 pool that most reduces log-loss.  Stops when no remaining feature
    improves log-loss over the current best.

    Returns a DataFrame with columns:
        step | feature_added | n_features | selected_features | xgb_ll | xgb_auc | delta_ll
    """
    suite = get_thesis_model_suite()
    all_features = suite["M13"]
    candidate_pool = [f for f in all_features if f in df.columns]

    selected: List[str] = []
    best_ll = float("inf")
    rows = []

    print(f"Greedy forward selection (M13 pool): {len(candidate_pool)} candidate features")

    step = 0
    while candidate_pool:
        best_feat: Optional[str] = None
        best_step_ll: Optional[float] = None
        best_step_auc: Optional[float] = None

        for feat in candidate_pool:
            trial = selected + [feat]
            ll, auc = _eval_features_xgb_cv(df, trial, group_col, n_splits, random_state)
            if ll is not None and (best_step_ll is None or ll < best_step_ll):
                best_step_ll = ll
                best_step_auc = auc
                best_feat = feat

        if best_feat is None or best_step_ll is None or best_step_ll >= best_ll:
            print(f"  Stopping at step {step}: no improvement (best_ll={best_ll:.4f})")
            break

        step += 1
        delta_ll = best_step_ll - best_ll if best_ll != float("inf") else None
        best_ll = best_step_ll
        selected.append(best_feat)
        candidate_pool.remove(best_feat)

        rows.append({
            "step": step,
            "feature_added": best_feat,
            "n_features": len(selected),
            "selected_features": ",".join(selected),
            "xgb_ll": best_step_ll,
            "xgb_auc": best_step_auc,
            "delta_ll": round(delta_ll, 4) if delta_ll is not None else None,
        })
        print(f"  Step {step}: +{best_feat} → ll={best_step_ll:.4f}, auc={best_step_auc:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Individual (leave-one-out) feature ablation
# ---------------------------------------------------------------------------

def run_single_feature_ablation(
    df: pd.DataFrame,
    base_model: str = "M12",
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-one-out individual feature ablation from the base_model feature set.

    Removes one feature at a time from the full model and re-evaluates with the
    same GroupKFold CV setup.  The first row is the full-model baseline.

    Returns a DataFrame sorted by delta_auc ascending (largest drop first):
        feature_removed | n_features_remaining | xgb_ll | xgb_auc | delta_ll | delta_auc
    """
    suite = get_thesis_model_suite()
    base_features = [f for f in suite[base_model] if f in df.columns]

    print(f"Individual ablation from {base_model} ({len(base_features)} available features)")

    # Baseline
    baseline_ll, baseline_auc = _eval_features_xgb_cv(
        df, base_features, group_col, n_splits, random_state
    )
    rows = [{
        "feature_removed": "—",
        "n_features_remaining": len(base_features),
        "xgb_ll": baseline_ll,
        "xgb_auc": baseline_auc,
        "delta_ll": 0.0,
        "delta_auc": 0.0,
    }]
    print(f"  Baseline ({base_model}): ll={baseline_ll:.4f}, auc={baseline_auc:.4f}")

    ablation_rows = []
    for feat in base_features:
        reduced = [f for f in base_features if f != feat]
        ll, auc = _eval_features_xgb_cv(df, reduced, group_col, n_splits, random_state)
        d_ll = round(ll - baseline_ll, 4) if ll is not None and baseline_ll is not None else None
        d_auc = round(auc - baseline_auc, 4) if auc is not None and baseline_auc is not None else None
        ablation_rows.append({
            "feature_removed": feat,
            "n_features_remaining": len(reduced),
            "xgb_ll": ll,
            "xgb_auc": auc,
            "delta_ll": d_ll,
            "delta_auc": d_auc,
        })
        print(f"  -{feat}: ll={ll:.4f}, auc={auc:.4f} (Δauc={d_auc:+.4f})")

    # Sort ablation rows by delta_auc ascending (largest drop = most important first)
    ablation_rows.sort(key=lambda r: r["delta_auc"] if r["delta_auc"] is not None else 0)
    return pd.DataFrame(rows + ablation_rows)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def run_shap_importance(
    df: pd.DataFrame,
    base_model: str = "M13",
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
    validate_bottom_k: int = 2,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Compute SHAP feature importance for the base_model feature set.

    Trains XGBoost on all data (no CV — for ranking only) and computes mean
    absolute SHAP values.  Optionally re-evaluates the model after removing
    the ``validate_bottom_k`` least important features to confirm minimal impact.

    Returns:
        shap_df   — DataFrame with columns: rank | feature | mean_abs_shap
        valid_df  — DataFrame with validation result, or None if validate_bottom_k <= 0
    """
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError(
            "shap is required for SHAP importance. Install with: pip install 'shap>=0.45'"
        )
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        raise ImportError("xgboost is required for SHAP importance.")

    suite = get_thesis_model_suite()
    base_features = [f for f in suite[base_model] if f in df.columns]
    print(f"SHAP importance on {base_model} ({len(base_features)} available features)")

    y = df["is_goal"].astype(int).to_numpy()
    X_full = _encode_features(df[base_features])

    # Impute NaNs with column medians (consistent with CV pipeline)
    medians = np.nanmedian(X_full, axis=0)
    for j in range(X_full.shape[1]):
        X_full[np.isnan(X_full[:, j]), j] = medians[j]

    _n_neg = int(np.sum(y == 0))
    _n_pos = int(np.sum(y == 1))
    _spw = _n_neg / max(_n_pos, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=_spw,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_full, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    # For binary XGBoost, shap_values may be 2D (n_samples × n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    mean_abs = np.abs(sv).mean(axis=0)

    shap_rows = sorted(
        [{"feature": f, "mean_abs_shap": round(float(v), 6)}
         for f, v in zip(base_features, mean_abs)],
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    for i, r in enumerate(shap_rows, 1):
        r["rank"] = i
        print(f"  {i:2d}. {r['feature']:30s}  mean|SHAP|={r['mean_abs_shap']:.6f}")

    shap_df = pd.DataFrame(shap_rows)[["rank", "feature", "mean_abs_shap"]]

    # Optional: re-evaluate after removing the bottom-k features
    valid_df: Optional[pd.DataFrame] = None
    if validate_bottom_k > 0 and len(shap_rows) > validate_bottom_k:
        bottom_feats = [r["feature"] for r in shap_rows[-validate_bottom_k:]]
        reduced_feats = [f for f in base_features if f not in bottom_feats]

        baseline_ll, baseline_auc = _eval_features_xgb_cv(
            df, base_features, group_col, n_splits, random_state
        )
        reduced_ll, reduced_auc = _eval_features_xgb_cv(
            df, reduced_feats, group_col, n_splits, random_state
        )
        d_ll = round(reduced_ll - baseline_ll, 4) if reduced_ll is not None and baseline_ll is not None else None
        d_auc = round(reduced_auc - baseline_auc, 4) if reduced_auc is not None and baseline_auc is not None else None
        valid_df = pd.DataFrame([
            {
                "variant": f"{base_model} full",
                "features_removed": "—",
                "xgb_ll": baseline_ll,
                "xgb_auc": baseline_auc,
                "delta_ll": 0.0,
                "delta_auc": 0.0,
            },
            {
                "variant": f"{base_model} −bottom{validate_bottom_k}",
                "features_removed": ",".join(bottom_feats),
                "xgb_ll": reduced_ll,
                "xgb_auc": reduced_auc,
                "delta_ll": d_ll,
                "delta_auc": d_auc,
            },
        ])
        print(f"  Validation (remove bottom {validate_bottom_k}): ll={reduced_ll:.4f} (Δ={d_ll:+.4f}), "
              f"auc={reduced_auc:.4f} (Δ={d_auc:+.4f})")

    return shap_df, valid_df


# ---------------------------------------------------------------------------
# MLP benchmark
# ---------------------------------------------------------------------------

def run_mlp_benchmark(
    df: pd.DataFrame,
    model_name: str = "M9",
    group_col: str = "game_id",
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate a simple MLP (sklearn) on one model configuration using GroupKFold CV.

    Uses the same pipeline as run_model_suite():
      - Per-fold StandardScaler
      - Sample weights via compute_sample_weight('balanced')
      - Pooled AUC and log-loss across folds
      - Train AUC computed per fold for overfitting diagnostics

    Returns a one-row DataFrame with columns:
      model, features, n_rows, mlp_ll, mlp_auc, mlp_train_ll, mlp_train_auc
    """
    from sklearn.neural_network import MLPClassifier

    suite = get_thesis_model_suite()
    if model_name not in suite:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(suite)}")

    all_features = suite[model_name]
    available = [f for f in all_features if f in df.columns]
    if len(available) < len(all_features):
        missing = set(all_features) - set(available)
        print(f"note: {len(missing)} feature(s) missing from data — {missing}")

    y = df["is_goal"].astype(int).to_numpy()
    X_raw = _encode_features(df[available])  # consistent with run_model_suite; NaNs preserved

    if group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        print(f"warning: group_col '{group_col}' not found — using sequential groups")
        groups = np.arange(len(df))

    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=actual_splits)

    pool_y_test: List[int] = []
    pool_p_test: List[float] = []
    pool_y_train: List[int] = []
    pool_p_train: List[float] = []

    for train_idx, test_idx in cv.split(X_raw, y, groups=groups):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr, X_te = _impute_with_train_median(X_tr, X_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # MLPClassifier does not support sample_weight in fit().
        # Class imbalance is handled implicitly; for a fair comparison, results
        # should be interpreted alongside XGBoost (scale_pos_weight) and LogReg
        # (class_weight='balanced').
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state,
        )
        mlp.fit(X_tr_s, y_tr)

        p_test = mlp.predict_proba(X_te_s)[:, 1]
        pool_y_test.extend(y_te.tolist())
        pool_p_test.extend(p_test.tolist())

        p_train = mlp.predict_proba(X_tr_s)[:, 1]
        pool_y_train.extend(y_tr.tolist())
        pool_p_train.extend(p_train.tolist())

    return pd.DataFrame([{
        "model":         model_name,
        "features":      ",".join(available),
        "n_rows":        len(df),
        "mlp_ll":        _pooled_ll(pool_y_test, pool_p_test),
        "mlp_auc":       _pooled_auc(pool_y_test, pool_p_test),
        "mlp_train_ll":  _pooled_ll(pool_y_train, pool_p_train),
        "mlp_train_auc": _pooled_auc(pool_y_train, pool_p_train),
    }])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_shots(csv_paths: List[Path]) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError("No input CSVs provided.")

    dfs = []
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["__source_csv"] = str(p)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(all_df.columns)}")

    all_df = all_df.copy()
    all_df["is_goal"] = pd.to_numeric(all_df["is_goal"], errors="coerce").astype("Int64")
    all_df["distance_m"] = pd.to_numeric(all_df["distance_m"], errors="coerce")
    all_df["angle_rad"] = pd.to_numeric(all_df["angle_rad"], errors="coerce")

    mask = (
        all_df["is_goal"].isin([0, 1])
        & np.isfinite(all_df["distance_m"].to_numpy(dtype=float))
        & np.isfinite(all_df["angle_rad"].to_numpy(dtype=float))
    )
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"dropped_rows_nonfinite: {dropped}")
    return all_df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Docs-style single-model baseline (geometry only, body part, play type)
# ---------------------------------------------------------------------------

def prepare_docs_style_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a docs-style xG table:
      - ball_goal_distance
      - shot_angle (degrees)
      - is_by_foot (0/1)  — uses shot_body_part if body_part is absent
      - type_of_play       — uses play_pattern if type_of_play is absent
      - shot_outcome (0/1)
    """
    out = df.copy()
    out["ball_goal_distance"] = pd.to_numeric(out["distance_m"], errors="coerce")
    out["shot_angle"] = np.degrees(pd.to_numeric(out["angle_rad"], errors="coerce"))
    out["shot_outcome"] = pd.to_numeric(out["is_goal"], errors="coerce").astype("Int64")

    # Remove unrealistic long-distance shots.
    out = out.loc[out["ball_goal_distance"] <= 53].copy()

    # Body part: prefer explicit body_part string, fall back to shot_body_part (0/1).
    if "body_part" in out.columns:
        bp = out["body_part"].astype(str).str.lower()
        out["is_by_foot"] = bp.str.contains("foot", na=False).astype(int)
    elif "shot_body_part" in out.columns:
        # shot_body_part: 1 = head, 0 = foot (pipeline convention)
        out["is_by_foot"] = (
            pd.to_numeric(out["shot_body_part"], errors="coerce")
            .fillna(0)
            .apply(lambda v: 0 if v == 1 else 1)
            .astype(int)
        )
    elif "is_by_foot" in out.columns:
        out["is_by_foot"] = pd.to_numeric(out["is_by_foot"], errors="coerce").fillna(1).astype(int)
    else:
        print("warning: no body-part column found — defaulting is_by_foot=1")
        out["is_by_foot"] = 1

    # Play type: prefer type_of_play string, fall back to play_pattern.
    if "type_of_play" in out.columns:
        out["type_of_play"] = out["type_of_play"].astype(str)
    elif "play_pattern" in out.columns:
        # Normalise play_pattern values to the expected type_of_play vocabulary.
        pat_map = {
            "regular_play": "regular_play",
            "open_play": "regular_play",
            "from_corner": "corner_kick",
            "corner_kick": "corner_kick",
            "from_free_kick": "free_kick",
            "free_kick": "free_kick",
            "from_counter": "counter_attack",
            "counter_attack": "counter_attack",
            "penalty": "penalty",
        }
        out["type_of_play"] = (
            out["play_pattern"].astype(str).str.lower().map(pat_map).fillna("regular_play")
        )
    else:
        print("warning: no play-type column found — defaulting type_of_play='regular_play'")
        out["type_of_play"] = "regular_play"

    out = out.loc[out["shot_outcome"].isin([0, 1])].copy()
    out["shot_outcome"] = out["shot_outcome"].astype(int)
    out["is_by_foot"] = out["is_by_foot"].astype(int)
    return out.reset_index(drop=True)


def fit_docs_logit(
    data: pd.DataFrame,
    title: str,
    random_state: int = 42,
) -> Optional[Tuple[LogisticRegression, StandardScaler]]:
    if len(data) < 20:
        print(f"skip_model_{title}: too few rows ({len(data)})")
        return None
    y = data["shot_outcome"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        print(f"skip_model_{title}: only one class")
        return None

    X = data[["ball_goal_distance", "shot_angle"]].to_numpy(dtype=float)
    min_class_count = int(np.bincount(y).min())
    stratify = y if min_class_count >= 2 else None
    if stratify is None:
        print(f"note_{title}: minority class has {min_class_count} member(s) — skipping stratify")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver="liblinear", max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    brier = float(brier_score_loss(y_test, y_pred_proba))
    b0, b1 = model.coef_[0]
    print(f"{title}: n={len(data)}, brier={brier:.4f}, beta_distance={b0:.3f}, beta_angle={b1:.3f}")
    return model, scaler


def predict_with_model(
    model_pair: Optional[Tuple[LogisticRegression, StandardScaler]],
    x1: float,
    x2: float,
    fallback: float,
) -> float:
    if model_pair is None:
        return float(fallback)
    model, scaler = model_pair
    X = np.asarray([[x1, x2]], dtype=float)
    return float(model.predict_proba(scaler.transform(X))[0, 1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classic xG baseline and M1–M12 thesis model suite."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more shot-level CSVs (features_final or aligned shots).",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument(
        "--run_model_suite",
        action="store_true",
        help="Run the M1–M12 thesis model suite with GroupKFold CV instead of the"
             " docs-style single baseline.",
    )
    ap.add_argument(
        "--run_ablation",
        action="store_true",
        help="Run backward feature ablation on M9.",
    )
    ap.add_argument(
        "--run_mlp",
        action="store_true",
        help="Run MLP benchmark on M9 with GroupKFold CV.",
    )
    ap.add_argument(
        "--run_greedy_forward",
        action="store_true",
        help="Run greedy forward feature selection on M13 feature pool (all 23 features).",
    )
    ap.add_argument(
        "--run_single_ablation",
        action="store_true",
        help="Run individual leave-one-out feature ablation from M12.",
    )
    ap.add_argument(
        "--run_shap",
        action="store_true",
        help="Compute SHAP feature importance on M12 and optionally validate bottom features.",
    )
    ap.add_argument(
        "--group_col",
        type=str,
        default="game_id",
        help="Column to use as GroupKFold groups (default: game_id).",
    )
    ap.add_argument(
        "--n_splits",
        type=int,
        default=4,
        help="Number of GroupKFold splits (default: 4, matching the 4-match pilot).",
    )
    args = ap.parse_args()

    raw = load_shots(list(args.inputs))

    if args.run_model_suite:
        _mlp_flag = getattr(args, "run_mlp", False)
        print(f"Running M1–M13 model suite (group_col={args.group_col}, n_splits={args.n_splits}, mlp={_mlp_flag})")
        suite_df = run_model_suite(
            raw,
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
            run_mlp=_mlp_flag,
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        suite_df.to_csv(args.out_csv, index=False)
        print("saved:", args.out_csv)
        return

    if args.run_ablation:
        print("Running backward ablation on M9...")
        abl_df = run_backward_ablation(
            raw,
            base_model="M9",
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        abl_path = args.out_csv.parent / (args.out_csv.stem + "_ablation.csv")
        abl_path.parent.mkdir(parents=True, exist_ok=True)
        abl_df.to_csv(abl_path, index=False)
        print(f"saved: {abl_path}")
        print(abl_df.to_string(index=False))
        return

    if args.run_mlp:
        print(f"Running MLP benchmark on M9 (group_col={args.group_col}, n_splits={args.n_splits})")
        mlp_df = run_mlp_benchmark(
            raw,
            model_name="M9",
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        mlp_path = args.out_csv.parent / (args.out_csv.stem + "_mlp.csv")
        mlp_path.parent.mkdir(parents=True, exist_ok=True)
        mlp_df.to_csv(mlp_path, index=False)
        print(f"saved: {mlp_path}")
        print(mlp_df.to_string(index=False))
        return

    if args.run_greedy_forward:
        print(f"Running greedy forward selection (group_col={args.group_col}, n_splits={args.n_splits})")
        fwd_df = run_greedy_forward_selection(
            raw,
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        fwd_path = args.out_csv.parent / (args.out_csv.stem + "_greedy_forward.csv")
        fwd_path.parent.mkdir(parents=True, exist_ok=True)
        fwd_df.to_csv(fwd_path, index=False)
        print(f"saved: {fwd_path}")
        print(fwd_df.to_string(index=False))
        return

    if args.run_single_ablation:
        print(f"Running individual feature ablation from M12 (20-feature pool; group_col={args.group_col}, n_splits={args.n_splits})")
        abl1_df = run_single_feature_ablation(
            raw,
            base_model="M12",
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        abl1_path = args.out_csv.parent / (args.out_csv.stem + "_single_ablation.csv")
        abl1_path.parent.mkdir(parents=True, exist_ok=True)
        abl1_df.to_csv(abl1_path, index=False)
        print(f"saved: {abl1_path}")
        print(abl1_df.to_string(index=False))
        return

    if args.run_shap:
        print(f"Running SHAP importance on M13")
        shap_df, valid_df = run_shap_importance(
            raw,
            base_model="M13",
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        shap_path = args.out_csv.parent / (args.out_csv.stem + "_shap_importance.csv")
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_path, index=False)
        print(f"saved: {shap_path}")
        print(shap_df.to_string(index=False))
        if valid_df is not None:
            valid_path = args.out_csv.parent / (args.out_csv.stem + "_shap_validation.csv")
            valid_df.to_csv(valid_path, index=False)
            print(f"saved: {valid_path}")
            print(valid_df.to_string(index=False))
        return

    # --- Default: docs-style geometry baseline ---
    data = prepare_docs_style_data(raw)

    print("n_rows_after_docs_preprocess:", len(data))
    print("n_goals:", int(data["shot_outcome"].sum()))

    open_play_data = data.loc[data["type_of_play"].isin(OPEN_PLAY_TYPES)].copy()
    free_kick_data = data.loc[data["type_of_play"] == "free_kick"].copy()
    penalty_data = data.loc[data["type_of_play"] == "penalty"].copy()
    foot_data = open_play_data.loc[open_play_data["is_by_foot"] == 1].copy()
    header_data = open_play_data.loc[open_play_data["is_by_foot"] == 0].copy()
    all_shots = pd.concat([foot_data, header_data, free_kick_data], ignore_index=True)

    penalty_xg = (
        float((penalty_data["shot_outcome"] == 1).mean())
        if len(penalty_data) > 0
        else float(data["shot_outcome"].mean())
    )
    print(f"penalty_xg_empirical: {penalty_xg:.3f} (n={len(penalty_data)})")

    models: Dict[str, Optional[Tuple[LogisticRegression, StandardScaler]]] = {
        "foot": fit_docs_logit(foot_data, "shots_by_foot", random_state=int(args.random_state)),
        "header": fit_docs_logit(header_data, "shots_by_header", random_state=int(args.random_state)),
        "free_kick": fit_docs_logit(free_kick_data, "shots_by_free_kick", random_state=int(args.random_state)),
        "all": fit_docs_logit(all_shots, "shots_all_combined", random_state=int(args.random_state)),
    }

    global_base = float(data["shot_outcome"].mean()) if len(data) > 0 else 0.0
    preds: List[float] = []
    model_used: List[str] = []

    for _, row in data.iterrows():
        dist = float(row["ball_goal_distance"])
        ang = float(row["shot_angle"])
        top = str(row["type_of_play"])
        by_foot = int(row["is_by_foot"])

        if top == "penalty":
            preds.append(float(penalty_xg))
            model_used.append("penalty_empirical")
            continue
        if top == "free_kick":
            preds.append(predict_with_model(models["free_kick"], dist, ang, fallback=global_base))
            model_used.append("free_kick")
            continue
        if top in OPEN_PLAY_TYPES and by_foot == 1:
            preds.append(predict_with_model(models["foot"], dist, ang, fallback=global_base))
            model_used.append("foot")
            continue
        if top in OPEN_PLAY_TYPES and by_foot == 0:
            preds.append(predict_with_model(models["header"], dist, ang, fallback=global_base))
            model_used.append("header")
            continue

        preds.append(predict_with_model(models["all"], dist, ang, fallback=global_base))
        model_used.append("all_fallback")

    data["xg_docs_style"] = pd.Series(preds, index=data.index, dtype=float).clip(0.0, 1.0)
    data["xg_docs_model_used"] = model_used
    data["xg_classic"] = data["xg_docs_style"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()
