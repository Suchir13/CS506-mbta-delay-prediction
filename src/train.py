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
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/processed/features.csv"
MODEL_OUT = "data/processed/best_model.pkl"

# Train / Val / Test split proportions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test gets the remaining ~15%


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
    df = df.sort_values(by=["day_of_week", "hour"]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    target = "is_delayed"
    feature_cols = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak",
    "route_encoded",
    "route_avg_delay",
    "is_rainy",
    "is_snowy",
    "TMAX",
    "TMIN",
    "PRCP",
    "SNOW",
    "AWND",
]

    X = df[feature_cols].values
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


def evaluate(name, model, X, y):
    """Print classification metrics for a fitted model."""
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # Use custom threshold instead of default 0.5
    threshold = 0.3
    preds = (proba >= threshold).astype(int) if proba is not None else model.predict(X)
    

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, proba) if proba is not None and len(np.unique(y)) > 1 else None

    print(f"\n  {name}")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1:        {f1:.3f}")
    if auc:
        print(f"    ROC-AUC:   {auc:.3f}")
    return f1  # Use F1 to pick best model


def train_models():
    df = load_features()

    if len(df) < 50:
        print("WARNING: Very few rows — results may not be meaningful.")

    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df)
    X_train_s, X_val_s, X_test_s, scaler = scale(X_train, X_val, X_test)

    models = {
        "Logistic Regression (baseline)": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print("\n=== Validation Set Results ===")
    best_name, best_model, best_f1 = None, None, -1

    results = []
    for name, model in models.items():
        # Logistic Regression benefits from scaling; tree models don't need it
        X_tr = X_train_s if "Logistic" in name else X_train
        X_v = X_val_s if "Logistic" in name else X_val

        model.fit(X_tr, y_train)
        f1 = evaluate(name, model, X_v, y_val)

        results.append({"model": name, "f1": round(f1, 3)})

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/processed/model_results.csv", index=False)
    print("\nSaved model results to data/processed/model_results.csv")

    print(f"\nBest model on validation set: {best_name} (F1={best_f1:.3f})")

    # Save validation predictions using best model
    X_v_best = X_val_s if "Logistic" in best_name else X_val
    proba_val = best_model.predict_proba(X_v_best)[:, 1] if hasattr(best_model, "predict_proba") else None

    threshold = 0.3
    preds_val = (proba_val >= threshold).astype(int) if proba_val is not None else best_model.predict(X_v_best)

    val_df = pd.DataFrame({
        "y_true": y_val,
        "y_pred": preds_val,
        "y_proba": proba_val if proba_val is not None else np.nan
    })

    val_df.to_csv("data/processed/val_predictions.csv", index=False)
    print("Saved validation predictions to data/processed/val_predictions.csv")

    # Final evaluation on held-out test set
    print("\n=== Test Set Results (best model) ===")
    X_t = X_test_s if "Logistic" in best_name else X_test
    evaluate(best_name, best_model, X_t, y_test)

    # Save predictions for evaluation
    proba = best_model.predict_proba(X_t)[:, 1] if hasattr(best_model, "predict_proba") else None
    threshold = 0.3
    preds = (proba >= threshold).astype(int) if proba is not None else best_model.predict(X_t)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": preds,
        "y_proba": proba if proba is not None else np.nan
    })

    pred_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("\nSaved test predictions to data/processed/test_predictions.csv")

    # Feature importance (for tree-based models)
    feature_cols = [c for c in df.columns if c != "is_delayed"]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        print("\n  Feature Importances (top 10):")
        print(importances.sort_values(ascending=False).head(10).to_string())

    # Save best model + scaler
    os.makedirs("data/processed", exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "features": feature_cols}, f)
    print(f"\nSaved best model to {MODEL_OUT}")

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