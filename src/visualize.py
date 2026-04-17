"""
visualize.py
------------
Generates exploratory and result visualizations:
  1. Delay rate by hour of day
  2. Delay rate by route
  3. Delay vs precipitation scatter
  4. Confusion matrix (best model)
  5. Feature importance bar chart

All plots saved to data/processed/plots/.

Run: python src/visualize.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLEAN_PATH = "data/processed/clean.csv"
FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "data/processed/best_model.pkl"
PLOTS_DIR = "data/processed/plots"
MAX_PRECIP_PLOT_ROWS = 100000

# Use a clean, readable style
sns.set_theme(style="whitegrid", palette="muted")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_clean_for_plots():
    """Load only the clean-data columns needed for current plots."""
    needed_cols = {"route_id", "hour", "is_delayed", "PRCP", "delay_minutes", "is_outlier"}
    return pd.read_csv(CLEAN_PATH, usecols=lambda c: c in needed_cols, low_memory=False)


def load_features_for_plots(feature_cols):
    """Load only the feature columns needed for model-result plots."""
    needed_cols = set(feature_cols) | {"is_delayed"}
    return pd.read_csv(FEATURES_PATH, usecols=lambda c: c in needed_cols, low_memory=False)


def plot_delay_by_hour(df):
    """Bar chart: % of buses delayed by hour of day."""
    if "hour" not in df.columns or "is_delayed" not in df.columns:
        print("Skipping hour plot: missing columns.")
        return

    hourly = df.groupby("hour")["is_delayed"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Bus Delay Rate by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 = midnight)")
    ax.set_ylabel("% of Arrivals Delayed > 5 min")
    ax.set_xticklabels(hourly.index, rotation=0)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/delay_by_hour.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_delay_by_route(df):
    """Horizontal bar chart: delay rate per route."""
    if "route_id" not in df.columns or "is_delayed" not in df.columns:
        print("Skipping route plot: missing columns.")
        return

    route_rate = (
        df.groupby("route_id")["is_delayed"]
        .mean()
        .sort_values(ascending=True) * 100
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(route_rate) * 0.5)))
    route_rate.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Bus Delay Rate by Route", fontsize=14, fontweight="bold")
    ax.set_xlabel("% of Arrivals Delayed > 5 min")
    ax.set_ylabel("Route ID")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/delay_by_route.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_delay_vs_precip(df):
    """Box plot: delay minutes on rainy vs non-rainy days."""
    if "PRCP" not in df.columns or "delay_minutes" not in df.columns:
        print("Skipping precipitation plot: missing columns.")
        return

    df_plot = df[~df["is_outlier"]].copy() if "is_outlier" in df.columns else df.copy()
    df_plot = df_plot[["PRCP", "delay_minutes"]].dropna().copy()

    if len(df_plot) > MAX_PRECIP_PLOT_ROWS:
        df_plot = df_plot.sample(MAX_PRECIP_PLOT_ROWS, random_state=42)
        print(f"Sampled {MAX_PRECIP_PLOT_ROWS:,} rows for precipitation plot.")

    dry = df_plot.loc[df_plot["PRCP"] <= 0.1, "delay_minutes"].to_numpy()
    rainy = df_plot.loc[df_plot["PRCP"] > 0.1, "delay_minutes"].to_numpy()

    if len(dry) == 0 or len(rainy) == 0:
        print("Skipping precipitation plot: insufficient dry/rainy samples.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([dry, rainy], labels=["Dry", 'Rainy (PRCP > 0.1")'], patch_artist=True)
    for patch, color in zip(ax.artists, ["skyblue", "lightcoral"]):
        patch.set_facecolor(color)
    ax.set_title("Delay Distribution: Rainy vs Dry Days", fontsize=14, fontweight="bold")
    ax.set_ylabel("Delay (minutes)")
    ax.set_ylim(-10, 30)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/delay_vs_precip.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(df_feat, model_bundle):
    """Confusion matrix on the test portion of the feature data."""
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["features"]

    target = "is_delayed"
    X = df_feat[feature_cols].values
    y = df_feat[target].values

    # Use last 15% as test set (mirrors train.py)
    test_start = int(len(X) * 0.85)
    X_test = X[test_start:]
    y_test = y[test_start:]

    if len(X_test) == 0:
        print("Not enough data for confusion matrix.")
        return

    model_name = type(model).__name__
    if "Logistic" in model_name:
        X_test = scaler.transform(X_test)

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On Time", "Delayed"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(model_bundle):
    """Bar chart of feature importances (tree-based models only)."""
    model = model_bundle["model"]
    feature_cols = model_bundle["features"]

    if not hasattr(model, "feature_importances_"):
        print("Skipping feature importance: model has no feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(importances) * 0.4)))
    importances.plot(kind="barh", ax=ax, color="mediumseagreen", edgecolor="white")
    ax.set_title("Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/feature_importance.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def make_plots():
    ensure_plots_dir()

    # Load best model
    model_bundle = None
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)

    # Load clean data for EDA plots
    df_clean = None
    if os.path.exists(CLEAN_PATH):
        df_clean = load_clean_for_plots()
        print(f"Loaded clean data for plots: {len(df_clean)} rows\n")
    else:
        print(f"WARNING: {CLEAN_PATH} not found. Run clean_data.py first.")

    # Load feature matrix for model plots
    df_feat = None
    if os.path.exists(FEATURES_PATH) and model_bundle is not None:
        df_feat = load_features_for_plots(model_bundle["features"])

    # Generate plots
    if df_clean is not None:
        plot_delay_by_hour(df_clean)
        plot_delay_by_route(df_clean)
        plot_delay_vs_precip(df_clean)

    if df_feat is not None and model_bundle is not None:
        plot_confusion_matrix(df_feat, model_bundle)
        plot_feature_importance(model_bundle)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    print("=== Generating Visualizations ===\n")
    make_plots()
    print("\nDone!")