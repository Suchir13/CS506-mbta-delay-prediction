"""
randomForestParams.py
---------------------
Compare several Random Forest parameter settings on the existing feature matrix.

For each configuration this script:
  - uses the same time-based split as train.py
  - tunes the classification threshold on the validation set for F1
  - evaluates on validation and test sets
  - saves the fitted model bundle under data/processed/models/
  - records metrics, training time, and model size in a summary CSV

Run:
    python src/randomForestParams.py
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from train import load_features, time_split, find_best_threshold, FEATURE_COLS


MODELS_DIR = "data/processed/models"
RESULTS_PATH = os.path.join(MODELS_DIR, "random_forest_param_results.csv")


RF_CONFIGS = [
    {
        "name": "rf_small",
        "n_estimators": 20,
        "max_depth": 12,
        "min_samples_split": 200,
        "min_samples_leaf": 100,
        "max_features": "sqrt",
    },
    {
        "name": "rf_compact_current",
        "n_estimators": 30,
        "max_depth": 16,
        "min_samples_split": 100,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
    },
    {
        "name": "rf_medium",
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_split": 50,
        "min_samples_leaf": 25,
        "max_features": "sqrt",
    },
    {
        "name": "rf_larger_bounded",
        "n_estimators": 80,
        "max_depth": 24,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
]


def compute_metrics(y_true, y_proba, threshold):
    preds = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
    }


def print_metrics(label, metrics, threshold):
    print(f"\n  {label} (threshold={threshold:.2f})")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")
    if not pd.isna(metrics["roc_auc"]):
        print(f"    ROC-AUC:   {metrics['roc_auc']:.3f}")


def model_filename(config):
    return (
        f"{config['name']}_"
        f"n{config['n_estimators']}_"
        f"d{config['max_depth']}_"
        f"split{config['min_samples_split']}_"
        f"leaf{config['min_samples_leaf']}.pkl"
    )


def train_and_compare_random_forests():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_features()
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df)

    all_results = []

    print("\n=== Random Forest Parameter Comparison ===")
    for idx, config in enumerate(RF_CONFIGS, start=1):
        print(f"\n[{idx}/{len(RF_CONFIGS)}] Training {config['name']}...")
        print(
            "  Params: "
            f"n_estimators={config['n_estimators']}, "
            f"max_depth={config['max_depth']}, "
            f"min_samples_split={config['min_samples_split']}, "
            f"min_samples_leaf={config['min_samples_leaf']}, "
            f"max_features={config['max_features']}"
        )

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        start = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - start

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, _ = find_best_threshold(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold)
        print_metrics(f"{config['name']} - validation", val_metrics, threshold)

        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, threshold)
        print_metrics(f"{config['name']} - test", test_metrics, threshold)

        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        print("  Top feature importances:")
        print(importances.sort_values(ascending=False).head(5).to_string())

        output_path = os.path.join(MODELS_DIR, model_filename(config))
        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLS,
                "threshold": threshold,
                "params": config,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            output_path,
            compress=3,
        )
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved model to {output_path} ({model_size_mb:.2f} MB)")

        all_results.append(
            {
                "name": config["name"],
                "model_path": output_path,
                "train_seconds": round(train_seconds, 2),
                "model_size_mb": round(model_size_mb, 2),
                "threshold": threshold,
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "min_samples_split": config["min_samples_split"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"],
                "val_accuracy": round(val_metrics["accuracy"], 3),
                "val_precision": round(val_metrics["precision"], 3),
                "val_recall": round(val_metrics["recall"], 3),
                "val_f1": round(val_metrics["f1"], 3),
                "val_roc_auc": round(val_metrics["roc_auc"], 3),
                "test_accuracy": round(test_metrics["accuracy"], 3),
                "test_precision": round(test_metrics["precision"], 3),
                "test_recall": round(test_metrics["recall"], 3),
                "test_f1": round(test_metrics["f1"], 3),
                "test_roc_auc": round(test_metrics["roc_auc"], 3),
            }
        )

    results_df = pd.DataFrame(all_results).sort_values(
        by=["val_f1", "test_f1", "model_size_mb"],
        ascending=[False, False, True],
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved comparison results to {RESULTS_PATH}")
    print("\n=== Comparison Summary (sorted by validation F1) ===")
    print(
        results_df[
            [
                "name",
                "model_size_mb",
                "train_seconds",
                "threshold",
                "val_f1",
                "val_roc_auc",
                "test_f1",
                "test_roc_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    train_and_compare_random_forests()
