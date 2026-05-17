"""
plot_results.py — Generate thesis result figures.

Usage:
    python code/xg_baseline/plot_results.py \
        --suite_csv   code/xg_baseline/outputs/classic_xg/all_matches_model_suite_weighted.csv \
        --ablation_csv code/xg_baseline/outputs/selection/results_single_ablation.csv \
        --shap_csv    code/xg_baseline/outputs/selection/results_shap_importance.csv \
        --out_dir     thesis/figures

Generates four figures:
    fig_auc_progression.png   — AUC across M1–M12 per model class
    fig_ablation_deltaAUC.png — leave-one-out ablation ΔAUC per feature
    fig_shap_importance.png   — SHAP mean |value| per feature
    fig_auc_literature.png    — AUC comparison with prior xG literature
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style — applied to every figure
# ---------------------------------------------------------------------------

STYLE = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "black",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "legend.frameon":    False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
}

# Greyscale palette — thesis/important = black, secondary = dark grey, tertiary = light grey
C_BLACK     = "#000000"
C_DARK      = "#444444"
C_MID       = "#888888"
C_LIGHT     = "#bbbbbb"

LINE_STYLES = {
    "LogReg":  (C_LIGHT,  "--", "o"),
    "XGBoost": (C_BLACK,  "-",  "s"),
    "TabPFN":  (C_DARK,   "-.", "^"),
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 — AUC progression across M1–M12
# ---------------------------------------------------------------------------

def plot_auc_progression(suite_csv: Path, out_path: Path) -> None:
    _apply_style()
    df = pd.read_csv(suite_csv)

    # Keep only M1–M12 rows in order
    model_order = [f"M{i}" for i in range(1, 13)]
    df = df[df["model"].isin(model_order)].copy()
    df["_ord"] = df["model"].map({m: i for i, m in enumerate(model_order)})
    df = df.sort_values("_ord").reset_index(drop=True)

    labels = df["model"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))

    cols = {
        "LogReg":  ("logreg_auc",  *LINE_STYLES["LogReg"]),
        "XGBoost": ("xgb_auc",     *LINE_STYLES["XGBoost"]),
        "TabPFN":  ("tabpfn_auc",  *LINE_STYLES["TabPFN"]),
    }

    for name, (col, colour, ls, marker) in cols.items():
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").tolist()
        ax.plot(x, y, color=colour, linestyle=ls, marker=marker,
                markersize=5, linewidth=1.4, label=name)

    # Annotate XGBoost M3 peak
    if "xgb_auc" in df.columns:
        xgb_vals = pd.to_numeric(df["xgb_auc"], errors="coerce")
        peak_idx = int(xgb_vals.idxmax())
        peak_val = float(xgb_vals.iloc[peak_idx])
        ax.annotate(
            f"M3: {peak_val:.3f}",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx + 0.4, peak_val - 0.06),
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color=C_MID, lw=0.8),
            color=C_BLACK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model specification")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.55, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 — Individual ablation ΔAUC
# ---------------------------------------------------------------------------

def plot_ablation_delta(ablation_csv: Path, out_path: Path) -> None:
    _apply_style()
    df = pd.read_csv(ablation_csv)

    # Drop the baseline row (feature_removed == "—")
    df = df[df["feature_removed"] != "—"].copy()
    df["delta_auc"] = pd.to_numeric(df["delta_auc"], errors="coerce")
    df = df.dropna(subset=["delta_auc"])

    # Sort: most negative (most important) at top
    df = df.sort_values("delta_auc", ascending=True).reset_index(drop=True)

    labels = df["feature_removed"].str.replace("_", r"\_", regex=False).tolist()
    values = df["delta_auc"].tolist()
    y = np.arange(len(labels))

    # Colour by sign: negative = dark, positive = light grey
    colours = [C_BLACK if v < -0.01 else C_MID if v < 0 else C_LIGHT for v in values]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y, values, color=colours, height=0.65, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$\Delta$AUC (removal)")
    ax.set_xlim(min(values) * 1.15, max(values) * 1.5 + 0.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 — SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap_importance(shap_csv: Path, out_path: Path) -> None:
    _apply_style()
    df = pd.read_csv(shap_csv)
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").fillna(0)

    # Sort most → least important (most important at top → reverse for barh)
    df = df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

    labels = df["feature"].str.replace("_", r"\_", regex=False).tolist()
    values = df["mean_abs_shap"].tolist()
    y = np.arange(len(labels))

    # Shade by value: top features darker
    max_v = max(values) if max(values) > 0 else 1
    colours = [str(0.1 + 0.7 * (1 - v / max_v)) for v in values]  # greyscale 0.1–0.8

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y, values, color=colours, height=0.65, edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"Mean $|\mathrm{SHAP}|$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 — AUC comparison with prior literature
# ---------------------------------------------------------------------------

LITERATURE = [
    ("Anzer \\& Bauer (2021)",     0.79,   "literature"),
    ("Mead et al.\\ (2023)",        0.80,   "literature"),
    ("Singh (2025)",                0.878,  "literature"),
    ("\\c{C}avu\\c{s} \\& Biecek (2022)", 0.819, "literature"),
    ("This thesis --- M1",          0.6742, "thesis"),
    ("This thesis --- M3",          0.9232, "thesis"),
    ("This thesis --- M12",         0.8828, "thesis"),
]
SE_THESIS = 0.13


def plot_auc_literature(out_path: Path) -> None:
    _apply_style()

    # Clean labels for matplotlib (strip LaTeX for display)
    display_labels = [
        "Anzer & Bauer (2021)",
        "Mead et al. (2023)",
        "Singh (2025)",
        "Cavus & Biecek (2022)",
        "This thesis — M1",
        "This thesis — M3",
        "This thesis — M12",
    ]
    aucs   = [row[1] for row in LITERATURE]
    types  = [row[2] for row in LITERATURE]

    y = np.arange(len(display_labels))
    colours = [C_BLACK if t == "thesis" else C_MID for t in types]
    xerr    = [SE_THESIS if t == "thesis" else 0 for t in types]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (auc, colour, err) in enumerate(zip(aucs, colours, xerr)):
        if err > 0:
            ax.errorbar(auc, i, xerr=err, fmt="s", color=colour,
                        markersize=7, elinewidth=1.0, capsize=3,
                        capthick=1.0, ecolor=C_MID, zorder=3)
        else:
            ax.plot(auc, i, "o", color=colour, markersize=7, zorder=3)

    ax.axvline(0.80, color=C_LIGHT, linewidth=0.9, linestyle="--", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.55, 1.05)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MID,
               markersize=7, label="Prior literature"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BLACK,
               markersize=7, label="This thesis (SE\u22480.13)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thesis result figures.")
    ap.add_argument("--suite_csv",    type=Path, required=True,
                    help="Model suite results CSV (logreg_auc, xgb_auc, tabpfn_auc columns).")
    ap.add_argument("--ablation_csv", type=Path, required=True,
                    help="Single-feature ablation CSV from --run_single_ablation.")
    ap.add_argument("--shap_csv",     type=Path, required=True,
                    help="SHAP importance CSV from --run_shap.")
    ap.add_argument("--out_dir",      type=Path, default=Path("thesis/figures"),
                    help="Directory to save figures (default: thesis/figures).")
    args = ap.parse_args()

    plot_auc_progression(
        args.suite_csv,
        args.out_dir / "fig_auc_progression.png",
    )
    plot_ablation_delta(
        args.ablation_csv,
        args.out_dir / "fig_ablation_deltaAUC.png",
    )
    plot_shap_importance(
        args.shap_csv,
        args.out_dir / "fig_shap_importance.png",
    )
    plot_auc_literature(
        args.out_dir / "fig_auc_literature.png",
    )


if __name__ == "__main__":
    main()
