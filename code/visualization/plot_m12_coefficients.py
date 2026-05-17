from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _load_classic_xg_module(project_root: Path):
    mod_path = project_root / "code" / "xg_baseline" / "classic_xg.py"
    spec = importlib.util.spec_from_file_location("classic_xg", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prepare_design_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    work = df[features + ["is_goal"]].copy()
    work["is_goal"] = pd.to_numeric(work["is_goal"], errors="coerce")
    work = work.loc[work["is_goal"].isin([0, 1])].copy()
    y = work["is_goal"].astype(int)
    X = work[features].copy()

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    if cat_cols:
        for c in cat_cols:
            s = X[c].astype("string")
            mode_vals = s.mode(dropna=True)
            fill_val = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "unknown"
            X[c] = s.fillna(fill_val).astype(str)
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit M12 logistic model and plot standardized coefficients.")
    parser.add_argument(
        "--inputs_csv",
        type=Path,
        default=Path("code/data_pipeline/outputs/all_matches_features_final.csv"),
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=Path("thesis/figures/feature_importance_m12.png"),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/tables/m12_coefficients.csv"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    mod = _load_classic_xg_module(project_root)

    df = pd.read_csv(args.inputs_csv)
    df = mod._ensure_canonical_feature_aliases(df)
    features = mod.get_thesis_model_suite()["M12"]

    X, y = _prepare_design_matrix(df, features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_scaled, y.to_numpy())

    coef_df = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_[0]})
        .sort_values("coefficient")
        .reset_index(drop=True)
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.xlabel("Coefficient value (standardized features)")
    plt.ylabel("Feature")
    plt.title("Logistic regression coefficients (M12)")
    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300)
    print("saved_png:", args.out_png)
    print("saved_csv:", args.out_csv)


if __name__ == "__main__":
    main()

