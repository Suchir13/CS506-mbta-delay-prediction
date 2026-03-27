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


def _fail(msg):
    raise ValueError(f"Data validation failed: {msg}")


def validate_schema(df):
    """
    Validate required columns for supported MBTA input formats.

    Supported delay sources:
      - delay_minutes
      - (travel_time_sec and benchmark_travel_time_sec)
    """
    if df.empty:
        _fail("input dataframe is empty")

    base_required = ["route_id", "date"]
    missing_base = [c for c in base_required if c not in df.columns]
    if missing_base:
        _fail(f"missing required base columns: {missing_base}")

    has_delay_col = "delay_minutes" in df.columns
    has_perf_cols = {"travel_time_sec", "benchmark_travel_time_sec"}.issubset(df.columns)
    if not (has_delay_col or has_perf_cols):
        _fail(
            "need either 'delay_minutes' OR both "
            "'travel_time_sec' and 'benchmark_travel_time_sec'"
        )


def normalize_core_fields(df):
    """Normalize key field types and canonical date format."""
    for col in ["route_id", "trip_id", "stop_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # Normalize date to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def drop_missing_required_keys(df):
    """
    Drop rows with missing critical keys.

    Always required:
      route_id, date

    Required if present in schema:
      trip_id, stop_id, scheduled_arrival
    """
    keys = ["route_id", "date"]
    for optional_key in ["trip_id", "stop_id", "scheduled_arrival"]:
        if optional_key in df.columns:
            keys.append(optional_key)

    before = len(df)
    df = df.dropna(subset=keys)
    dropped = before - len(df)
    return df, keys, dropped


def deduplicate_mbta(df):
    """Apply deterministic deduplication with a stable keep-first rule."""
    dedup_key_candidates = ["route_id", "trip_id", "stop_id", "date", "scheduled_arrival"]
    dedup_keys = [c for c in dedup_key_candidates if c in df.columns]

    if len(dedup_keys) < 3:
        # Not enough reliable identifiers to deduplicate safely.
        return df, dedup_keys, 0

    before = len(df)
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    df = df.sort_values(by=dedup_keys + ["_row_order"], kind="mergesort")
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    df = df.sort_values(by="_row_order", kind="mergesort").drop(columns=["_row_order"])
    dropped = before - len(df)
    return df, dedup_keys, dropped


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

    validate_schema(df)
    raw_rows = len(df)

    df = normalize_core_fields(df)
    df, required_keys, dropped_missing = drop_missing_required_keys(df)

    df, dedup_keys, dropped_dupes = deduplicate_mbta(df)

    # Ensure delay_minutes column exists
    if "delay_minutes" not in df.columns:
        if "travel_time_sec" in df.columns and "benchmark_travel_time_sec" in df.columns:
            df["delay_minutes"] = (df["travel_time_sec"] - df["benchmark_travel_time_sec"]) / 60.0
        else:
            print("Cannot compute delay_minutes — missing required columns.")
            return pd.DataFrame()

    before_delay_drop = len(df)
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
    df = df.dropna(subset=["delay_minutes", "date"])
    dropped_invalid_delay = before_delay_drop - len(df)

    df["is_outlier"] = (df["delay_minutes"].abs() > OUTLIER_MINUTES)
    df["is_delayed"] = (df["delay_minutes"] > DELAY_THRESHOLD_MINUTES).astype(int)

    n_out = df["is_outlier"].sum()
    print("Cleaning summary:")
    print(f"  Raw rows: {raw_rows}")
    print(f"  Dropped missing required keys {required_keys}: {dropped_missing}")
    if dedup_keys:
        print(f"  Dedup key: {dedup_keys}")
        print(f"  Duplicates removed: {dropped_dupes}")
    else:
        print("  Dedup skipped: insufficient key columns")
    print(f"  Dropped invalid/non-numeric delay rows: {dropped_invalid_delay}")
    print(f"  Outlier rows flagged (|delay| > {OUTLIER_MINUTES} min): {n_out}")
    print(f"  Final cleaned rows: {len(df)}")

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