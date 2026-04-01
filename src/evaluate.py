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

    print("\nDone!")