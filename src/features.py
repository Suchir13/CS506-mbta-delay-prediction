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
    df = pd.read_csv(CLEAN_PATH, dtype={"route_id": str}, low_memory=False)
    df = df.copy()
    if "route_id" in df.columns:
        df.loc[:, "route_id"] = df["route_id"].astype(str).str.strip()
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
    df = df.copy()

    if "PRCP" in df.columns:
        df.loc[:, "is_rainy"] = (df["PRCP"] > 0.1).astype(int)
    else:
        df.loc[:, "is_rainy"] = 0

    if "SNOW" in df.columns:
        df.loc[:, "is_snowy"] = (df["SNOW"] > 0.1).astype(int)
    else:
        df.loc[:, "is_snowy"] = 0

    return df


def encode_route(df):
    """
    Label-encode route_id as a numeric feature.
    We use a simple mapping rather than one-hot to keep it manageable.
    """
    df = df.copy()
    df.loc[:, "route_id"] = df["route_id"].astype(str).str.strip()
    route_ids = sorted(df["route_id"].dropna().unique().tolist())
    route_map = {r: i for i, r in enumerate(route_ids)}
    df.loc[:, "route_encoded"] = df["route_id"].map(route_map)
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

    # New approach: Instead of dropping all rows with any NaN, we fill weather-related NaNs with defaults and only drop rows missing critical features.
    # Handle missing values (especially weather data)
    before = len(df_feat)

    # Fill weather-related columns with defaults
    weather_cols = ["TMAX", "TMIN", "PRCP", "SNOW", "AWND"]
    for col in weather_cols:
        if col in df_feat.columns:
            df_feat.loc[:, col] = df_feat[col].fillna(0)

    # Drop rows only if critical columns are missing
    df_feat = df_feat.dropna(subset=["hour", "day_of_week", "route_encoded", "is_delayed"])

    after = len(df_feat)
    #indented return backwards
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