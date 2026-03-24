"""
clean_data.py
-------------
Cleans MBTA travel time data and merges with weather.
Handles both Performance API output and schedule-fallback output.

Run: python src/clean_data.py
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTLIER_MINUTES = 120
DELAY_THRESHOLD_MINUTES = 5


def load_travel_times():
    path = f"{RAW_DIR}/travel_times.csv"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run collect_mbta.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} travel time rows.")
    return df


def load_weather():
    path = f"{RAW_DIR}/weather.csv"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run collect_weather.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} weather rows.")
    return df


def parse_time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds. Handles MBTA's >24hr format."""
    if pd.isna(time_str) or time_str == "":
        return np.nan
    try:
        parts = str(time_str).split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return np.nan


def add_time_features(df):
    """Extract hour, day of week, weekend, and peak hour flags."""
    if df.empty:
        return df

    # Get hour from scheduled_arrival if available
    if "scheduled_arrival" in df.columns:
        df["hour"] = df["scheduled_arrival"].apply(
            lambda t: int(str(t).split(":")[0]) % 24 if pd.notna(t) and ":" in str(t) else np.nan
        )
    elif "dep_dt" in df.columns:
        # From Performance API: dep_dt is a unix timestamp
        dep = pd.to_numeric(df["dep_dt"], errors="coerce")
        df["hour"] = pd.to_datetime(dep, unit="s", errors="coerce").dt.hour
    else:
        df["hour"] = np.nan

    df["day_of_week"] = pd.to_datetime(df["date"], errors="coerce").dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"] = (
        (~df["is_weekend"].astype(bool)) &
        (df["hour"].between(7, 9) | df["hour"].between(16, 19))
    ).astype(int)

    return df


def clean_mbta(df):
    """Remove bad rows, flag outliers, ensure delay and target columns exist."""
    if df.empty:
        return df

    # Ensure delay_minutes column exists
    if "delay_minutes" not in df.columns:
        if "travel_time_sec" in df.columns and "benchmark_travel_time_sec" in df.columns:
            df["delay_minutes"] = (df["travel_time_sec"] - df["benchmark_travel_time_sec"]) / 60.0
        else:
            print("Cannot compute delay_minutes — missing required columns.")
            return pd.DataFrame()

    df = df.dropna(subset=["delay_minutes", "date"])
    df["is_outlier"] = (df["delay_minutes"].abs() > OUTLIER_MINUTES)
    df["is_delayed"] = (df["delay_minutes"] > DELAY_THRESHOLD_MINUTES).astype(int)

    n_out = df["is_outlier"].sum()
    if n_out > 0:
        print(f"Flagged {n_out} outlier rows (|delay| > {OUTLIER_MINUTES} min).")

    return df


def merge_weather(df_mbta, df_weather):
    """Merge daily weather onto MBTA data by date."""
    if df_weather.empty:
        for col in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]:
            df_mbta[col] = np.nan
        return df_mbta

    df_mbta["date"] = df_mbta["date"].astype(str)
    df_weather["date"] = df_weather["date"].astype(str)
    df = pd.merge(df_mbta, df_weather, on="date", how="left")

    for col in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]:
        if col in df.columns:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"  Imputing {n_missing} missing {col} values with median ({median_val:.1f}).")
            df[col] = df[col].fillna(median_val)

    return df


def clean_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_travel_times()
    df_weather = load_weather()

    df = clean_mbta(df)
    if df.empty:
        print("No data to clean. Exiting.")
        return

    df = add_time_features(df)
    df = merge_weather(df, df_weather)

    final_cols = [
        "route_id", "date", "hour", "day_of_week", "is_weekend", "is_peak",
        "delay_minutes", "is_delayed", "is_outlier",
        "TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df_clean = df[final_cols]

    out_path = f"{PROCESSED_DIR}/clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nClean dataset: {len(df_clean)} rows, {df_clean['is_delayed'].mean()*100:.1f}% delayed")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    print("=== Cleaning Data ===\n")
    clean_data()
    print("\nDone!")