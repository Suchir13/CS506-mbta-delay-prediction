"""
train.py
--------
Trains and evaluates three classifiers on the feature matrix:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Gradient Boosted Trees

Uses a time-based train/validation/test split to avoid leakage.
Saves the best model to data/processed/best_model.pkl and prints metrics.

Run: python src/train.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

FEATURES_PATH = "data/processed/features.csv"
MODEL_OUT = "data/processed/best_model.pkl"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
RF_N_ESTIMATORS = 30
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_SPLIT = 100
RF_MIN_SAMPLES_LEAF = 50
RF_MAX_FEATURES = "sqrt"

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "is_peak",
    "route_encoded",
    # route_avg_delay excluded: computed on full dataset → data leakage
    "is_rainy", "is_snowy",
    "TMAX", "TMIN", "PRCP", "SNOW", "AWND",
]


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run features.py first.")
    
    df = pd.read_csv(FEATURES_PATH)

    print(f"\nLoaded feature matrix: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution (is_delayed):")
    print(df["is_delayed"].value_counts())

    return df


def time_split(df):
    """
    Split data in chronological order (no shuffle!) to avoid temporal leakage.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    
    # Sort data chronologically (approximate using day_of_week and hour)
    # No sort — preserve original chronological collection order

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    target = "is_delayed"
    X = df[FEATURE_COLS].values
    y = df[target].values

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print("\n=== Split Verification ===")
    print(f"Train indices: 0 → {train_end}")
    print(f"Val indices: {train_end} → {val_end}")
    print(f"Test indices: {val_end} → {n}")

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale(X_train, X_val, X_test):
    """Standardize features (mean=0, std=1). Fit ONLY on training data."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds on the validation set and return the one that maximises F1."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, round(float(t), 2)
    return best_thresh, best_f1


def evaluate(name, model, X, y, threshold=0.5):
    """Print classification metrics for a fitted model at a given threshold."""
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    preds = (proba >= threshold).astype(int) if proba is not None else model.predict(X)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    auc  = roc_auc_score(y, proba) if proba is not None and len(np.unique(y)) > 1 else None

    print(f"\n  {name}  (threshold={threshold:.2f})")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1:        {f1:.3f}")
    if auc:
        print(f"    ROC-AUC:   {auc:.3f}")
    return f1, proba


def train_models():
    df = load_features()

    if len(df) < 50:
        print("WARNING: Very few rows — results may not be meaningful.")

    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df)
    X_train_s, X_val_s, X_test_s, scaler = scale(X_train, X_val, X_test)

    # Class weights for imbalanced target
    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        # "Logistic Regression (baseline)": LogisticRegression(
        #     max_iter=1000, random_state=42, class_weight="balanced"
        # ),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        # "Gradient Boosted Trees": GradientBoostingClassifier(
        #     n_estimators=100, random_state=42
        # ),
    }

    print("\n=== Validation Set Results ===")
    best_name, best_model, best_f1, best_threshold = None, None, -1, 0.5

    results = []
    thresholds = {}
    for name, model in models.items():
        X_tr = X_train_s if "Logistic" in name else X_train
        X_v  = X_val_s   if "Logistic" in name else X_val

        # GBT uses sample_weight; LR and RF use class_weight
        if "Gradient" in name:
            model.fit(X_tr, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_tr, y_train)

        # Tune threshold on validation set
        proba_v = model.predict_proba(X_v)[:, 1]
        thresh, _ = find_best_threshold(y_val, proba_v)
        thresholds[name] = thresh

        f1, _ = evaluate(name, model, X_v, y_val, threshold=thresh)
        results.append({"model": name, "f1": round(f1, 3), "threshold": thresh})

        if f1 > best_f1:
            best_f1, best_name, best_model, best_threshold = f1, name, model, thresh

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/processed/model_results.csv", index=False)
    print("\nSaved model results to data/processed/model_results.csv")
    print(f"\nBest model on validation set: {best_name} (F1={best_f1:.3f}, threshold={best_threshold:.2f})")

    # Save validation predictions using best model + tuned threshold
    X_v_best  = X_val_s if "Logistic" in best_name else X_val
    proba_val = best_model.predict_proba(X_v_best)[:, 1]
    preds_val = (proba_val >= best_threshold).astype(int)

    val_df = pd.DataFrame({
        "y_true": y_val, "y_pred": preds_val, "y_proba": proba_val
    })
    val_df.to_csv("data/processed/val_predictions.csv", index=False)
    print("Saved validation predictions to data/processed/val_predictions.csv")

    # Final evaluation on held-out test set
    print("\n=== Test Set Results (best model) ===")
    X_t   = X_test_s if "Logistic" in best_name else X_test
    evaluate(best_name, best_model, X_t, y_test, threshold=best_threshold)

    proba = best_model.predict_proba(X_t)[:, 1]
    preds = (proba >= best_threshold).astype(int)

    pred_df = pd.DataFrame({
        "y_true": y_test, "y_pred": preds, "y_proba": proba
    })

    pred_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("\nSaved test predictions to data/processed/test_predictions.csv")

    # Feature importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        print("\n  Feature Importances (top 10):")
        print(importances.sort_values(ascending=False).head(10).to_string())

    # Save best model + scaler
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(
        {"model": best_model, "scaler": scaler, "features": FEATURE_COLS},
        MODEL_OUT,
        compress=3,
    )
    print(f"\nSaved compressed best model to {MODEL_OUT}")

    split_info = {
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test)
    }

    pd.Series(split_info).to_csv("data/processed/split_info.csv")
    print("Saved split info to data/processed/split_info.csv")


if __name__ == "__main__":
    print("=== Training Models ===\n")
    train_models()
    print("\nDone!")