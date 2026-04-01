import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os


TEST_PATH = "data/processed/test_predictions.csv"
VAL_PATH = "data/processed/val_predictions.csv"
FEATURES_PATH = "data/processed/features.csv"

def load_predictions():
    test_df = pd.read_csv(TEST_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print("Test shape:", test_df.shape)
    print("Validation shape:", val_df.shape)
    print("\nColumns:", test_df.columns.tolist())

    return test_df, val_df

def compute_metrics(df, name="Dataset"):
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    y_proba = df["y_proba"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    print(f"\n=== {name} Metrics ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}")

def compute_confusion(df, name="Dataset"):
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} Confusion Matrix ===")
    print(f"TN (Correct No Delay): {tn}")
    print(f"FP (False Alarm):      {fp}")
    print(f"FN (Missed Delay):     {fn}")
    print(f"TP (Correct Delay):    {tp}")

def plot_roc(df, name="Dataset"):
    y_true = df["y_true"]
    y_proba = df["y_proba"]

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.show()

def plot_pr(df, name="Dataset"):
    y_true = df["y_true"]
    y_proba = df["y_proba"]

    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    os.makedirs("data/processed/plots", exist_ok=True)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {name}")
    plt.savefig(f"data/processed/plots/pr_curve_{name.lower()}.png")
    plt.close()

    print(f"Saved PR curve → data/processed/plots/pr_curve_{name.lower()}.png")

def slice_peak(df):
    print("\n=== Peak vs Off-Peak ===")

    for label, subset in [("Peak", df[df["is_peak"] == 1]),
                          ("Off-Peak", df[df["is_peak"] == 0])]:

        if len(subset) == 0:
            continue

        acc = accuracy_score(subset["y_true"], subset["y_pred"])
        f1 = f1_score(subset["y_true"], subset["y_pred"], zero_division=0)

        print(f"{label} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

def slice_route(df):
    print("\n=== Route-wise Performance ===")

    for route in df["route_encoded"].unique()[:5]:  # limit to 5 routes
        subset = df[df["route_encoded"] == route]

        if len(subset) < 50:
            continue

        acc = accuracy_score(subset["y_true"], subset["y_pred"])
        f1 = f1_score(subset["y_true"], subset["y_pred"], zero_division=0)

        print(f"Route {route} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

def load_with_features(pred_path):
    pred_df = pd.read_csv(pred_path)
    feat_df = pd.read_csv(FEATURES_PATH)

    # Align rows (since split was sequential)
    merged_df = feat_df.iloc[-len(pred_df):].copy()
    merged_df["y_true"] = pred_df["y_true"].values
    merged_df["y_pred"] = pred_df["y_pred"].values
    merged_df["y_proba"] = pred_df["y_proba"].values

    return merged_df

if __name__ == "__main__":
    print("=== Loading Predictions ===\n")
    test_df, val_df = load_predictions()

    compute_metrics(val_df, "Validation")
    compute_confusion(val_df, "Validation")

    compute_metrics(test_df, "Test")
    compute_confusion(test_df, "Test")

    plot_roc(test_df, "Test")
    plot_pr(val_df, "Validation")
    plot_pr(test_df, "Test")

    test_df = pd.read_csv(TEST_PATH)
    test_full = load_with_features(TEST_PATH)

    slice_peak(test_full)
    slice_route(test_full)

    print("\nDone!")