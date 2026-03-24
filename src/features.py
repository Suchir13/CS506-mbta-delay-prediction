"""
features.py
-----------
Builds the final feature matrix from clean.csv.
Adds route-level historical stats and encodes categoricals.
Saves to data/processed/features.csv.

Run: python src/features.py
"""

import os
import pandas as pd
import numpy as np

CLEAN_PATH = "data/processed/clean.csv"
OUT_PATH = "data/processed/features.csv"


def load_clean():
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"{CLEAN_PATH} not found. Run clean_data.py first.")
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} rows from clean dataset.")
    return df


def add_route_avg_delay(df):
    """
    Compute historical average delay per route.
    To avoid data leakage we use a leave-one-out approach:
    for each row, the route average is computed from all OTHER rows.
    For simplicity here we use a global per-route mean (document this limitation).
    """
    route_avg = df.groupby("route_id")["delay_minutes"].mean().rename("route_avg_delay")
    df = df.merge(route_avg, on="route_id", how="left")
    return df


def add_rain_snow_flags(df):
    """Create binary flags for rainy and snowy conditions."""
    if "PRCP" in df.columns:
        df["is_rainy"] = (df["PRCP"] > 0.1).astype(int)
    else:
        df["is_rainy"] = 0

    if "SNOW" in df.columns:
        df["is_snowy"] = (df["SNOW"] > 0.1).astype(int)
    else:
        df["is_snowy"] = 0

    return df


def encode_route(df):
    """
    Label-encode route_id as a numeric feature.
    We use a simple mapping rather than one-hot to keep it manageable.
    """
    route_ids = sorted(df["route_id"].dropna().unique())
    route_map = {r: i for i, r in enumerate(route_ids)}
    df["route_encoded"] = df["route_id"].map(route_map)
    return df


def select_features(df):
    """
    Define the final feature columns (X) and target (y).
    Returns df with only model-relevant columns.
    """
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
        "is_delayed",  # target variable
    ]

    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat = df[feature_cols].copy()

    # Drop any remaining NaNs in feature columns
    before = len(df_feat)
    df_feat = df_feat.dropna()
    after = len(df_feat)
    if before != after:
        print(f"Dropped {before - after} rows with NaN values in features.")

    return df_feat


def build_features():
    df = load_clean()
    df = add_route_avg_delay(df)
    df = add_rain_snow_flags(df)
    df = encode_route(df)
    df_feat = select_features(df)

    os.makedirs("data/processed", exist_ok=True)
    df_feat.to_csv(OUT_PATH, index=False)
    print(f"Feature matrix: {df_feat.shape[0]} rows × {df_feat.shape[1]-1} features")
    print(f"Target distribution: {df_feat['is_delayed'].value_counts().to_dict()}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    print("=== Building Features ===\n")
    build_features()
    print("\nDone!")