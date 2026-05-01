"""
clean_data.py
-------------
Cleans MBTA travel time data from MassGIS official arrival/departure CSVs
and merges with weather data.

Data source:
  MassGIS official MBTA arrival/departure monthly CSVs in data/raw/arrival_departure/
  Download from:
    2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
    2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c

Run:
    python src/clean_data.py
    python src/clean_data.py --dataset-dir data/raw/arrival_departure
    python src/clean_data.py --official-standard-type all
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OFFICIAL_DATASET_DIR = f"{RAW_DIR}/arrival_departure"
OUTLIER_MINUTES = 120
DELAY_THRESHOLD_MINUTES = 5
OFFICIAL_STANDARD_TYPE_DEFAULT = "schedule"


def print_progress_bar(current, total, prefix="Progress", width=30):
    """Print a simple in-place progress bar."""
    total = max(total, 1)
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}: [{bar}] {ratio * 100:5.1f}%", end="", flush=True)
    if current >= total:
        print()


def _fail(msg):
    raise ValueError(f"Data validation failed: {msg}")


def validate_schema(df):
    """
    Validate required columns for MassGIS MBTA input format.

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
    df = df.copy()
    for col in ["route_id", "trip_id", "stop_id"]:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(str).str.strip()
            df.loc[:, col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # Normalize date to YYYY-MM-DD
    df.loc[:, "date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
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
        return df, dedup_keys, 0

    before = len(df)
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    df = df.sort_values(by=dedup_keys + ["_row_order"], kind="mergesort")
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    df = df.sort_values(by="_row_order", kind="mergesort").drop(columns=["_row_order"])
    dropped = before - len(df)
    return df, dedup_keys, dropped


def official_timestamp_series_to_service_hms(series):
    """Vectorized conversion of MassGIS timestamps to service-day HH:MM:SS.
    Example: 1900-01-02T04:57:00Z -> 28:57:00
    """
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    hours = (ts.dt.hour + 24 * (ts.dt.day - 1)).astype("Int64")
    minutes = ts.dt.minute.astype("Int64")
    seconds = ts.dt.second.astype("Int64")

    out = pd.Series(np.nan, index=series.index, dtype="object")
    valid = ts.notna()
    out.loc[valid] = (
        hours.loc[valid].astype(str).str.zfill(2)
        + ":"
        + minutes.loc[valid].astype(str).str.zfill(2)
        + ":"
        + seconds.loc[valid].astype(str).str.zfill(2)
    )
    return out


def filter_official_standard_type(df, official_standard_type="schedule"):
    """
    Filter MassGIS rows by standard type.

    Modes:
      - schedule: keep only Schedule rows
      - headway: keep only Headway rows
      - all: keep both
    """
    if "standard_type" not in df.columns:
        return df

    mode = str(official_standard_type).strip().lower()
    if mode == "all":
        print(f"Keeping all official standard types: {len(df)} rows")
        return df

    expected = "Schedule" if mode == "schedule" else "Headway"
    before = len(df)
    df = df[df["standard_type"].astype(str).str.strip().eq(expected)].copy()
    print(f"Filtered to {expected} rows: {len(df)} kept ({before - len(df)} removed)")
    return df


def normalize_massgis_arrival_departure(df):
    """Normalize MassGIS MBTA arrival/departure rows into the pipeline schema."""
    scheduled_ts = pd.to_datetime(df["scheduled"], errors="coerce", utc=True)
    actual_ts = pd.to_datetime(df["actual"], errors="coerce", utc=True)

    print("Computing normalized delay fields...")
    df_norm = pd.DataFrame({
        "route_id": df["route_id"],
        "trip_id": df["half_trip_id"],
        "stop_id": df["stop_id"],
        "date": df["service_date"],
        "stop_sequence": df["time_point_order"],
        "scheduled_timestamp": df["scheduled"],
        "actual_timestamp": df["actual"],
        "scheduled_arrival": official_timestamp_series_to_service_hms(df["scheduled"]),
        "actual_arrival": official_timestamp_series_to_service_hms(df["actual"]),
        "has_actual": actual_ts.notna().astype(int),
        "delay_minutes": (actual_ts - scheduled_ts).dt.total_seconds() / 60.0,
        "is_delayed": ((actual_ts - scheduled_ts).dt.total_seconds() / 60.0 > DELAY_THRESHOLD_MINUTES).astype(int),
        # Preserve richer MassGIS fields for downstream use.
        "direction_id": df["direction_id"],
        "time_point_id": df["time_point_id"],
        "time_point_order": df["time_point_order"],
        "point_type": df["point_type"],
        "standard_type": df["standard_type"],
        "scheduled_headway": df["scheduled_headway"],
        "headway": df["headway"],
    })

    for col in ["time_point_order", "stop_sequence", "scheduled_headway", "headway", "delay_minutes"]:
        if col in df_norm.columns:
            df_norm.loc[:, col] = pd.to_numeric(df_norm[col], errors="coerce")

    return df_norm


def count_data_rows(path):
    """Count non-header rows in a CSV file for progress reporting."""
    with open(path, "rb") as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)


def read_csv_with_progress(path, usecols, dtype, chunksize=200000, prefix="Loading CSV"):
    """Read a CSV in chunks and show progress by rows loaded."""
    total_rows = count_data_rows(path)
    if total_rows == 0:
        return pd.DataFrame(columns=usecols)

    frames = []
    loaded_rows = 0
    print(f"{prefix}: {os.path.basename(path)} ({total_rows:,} rows)")
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
        chunksize=chunksize,
    ):
        frames.append(chunk)
        loaded_rows += len(chunk)
        print_progress_bar(loaded_rows, total_rows, prefix="  reading")

    return pd.concat(frames, ignore_index=True)


def load_massgis_arrival_departure(dataset_dir=OFFICIAL_DATASET_DIR, official_standard_type=OFFICIAL_STANDARD_TYPE_DEFAULT):
    """
    Load and normalize all MassGIS MBTA arrival/departure monthly CSVs.

    Download from:
      2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
      2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c
    Place files in data/raw/arrival_departure/ before running.
    """
    pattern = os.path.join(dataset_dir, "*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"WARNING: No MassGIS arrival/departure CSVs found in {dataset_dir}")
        print("Download monthly CSVs from MassGIS and place them in that directory.")
        return pd.DataFrame()

    usecols = [
        "service_date", "route_id", "direction_id", "half_trip_id", "stop_id",
        "time_point_id", "time_point_order", "point_type", "standard_type",
        "scheduled", "actual", "scheduled_headway", "headway",
    ]

    frames = []
    total_files = len(paths)
    for idx, path in enumerate(paths, 1):
        print_progress_bar(idx - 1, total_files, prefix="Loading MassGIS files")
        df_part = read_csv_with_progress(
            path,
            usecols=usecols,
            dtype={
                "route_id": str,
                "direction_id": str,
                "half_trip_id": str,
                "stop_id": str,
                "time_point_id": str,
                "point_type": str,
                "standard_type": str,
            },
            prefix="Loading file",
        )
        frames.append(df_part)
    print_progress_bar(total_files, total_files, prefix="Loading MassGIS files")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} MassGIS file(s).")

    df = filter_official_standard_type(df, official_standard_type=official_standard_type)
    return normalize_massgis_arrival_departure(df)


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

    df = df.copy()

    if "scheduled_arrival" in df.columns:
        df.loc[:, "hour"] = df["scheduled_arrival"].apply(
            lambda t: int(str(t).split(":")[0]) % 24 if pd.notna(t) and ":" in str(t) else np.nan
        )
    else:
        df.loc[:, "hour"] = np.nan

    df.loc[:, "day_of_week"] = pd.to_datetime(df["date"], errors="coerce").dt.dayofweek
    df.loc[:, "is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df.loc[:, "is_peak"] = (
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

    if "delay_minutes" not in df.columns:
        if "travel_time_sec" in df.columns and "benchmark_travel_time_sec" in df.columns:
            df["delay_minutes"] = (df["travel_time_sec"] - df["benchmark_travel_time_sec"]) / 60.0
        else:
            print("Cannot compute delay_minutes — missing required columns.")
            return pd.DataFrame()

    before_delay_drop = len(df)
    df.loc[:, "delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
    df = df.dropna(subset=["delay_minutes", "date"]).copy()
    dropped_invalid_delay = before_delay_drop - len(df)

    df.loc[:, "is_outlier"] = (df["delay_minutes"].abs() > OUTLIER_MINUTES)
    df.loc[:, "is_delayed"] = (df["delay_minutes"] > DELAY_THRESHOLD_MINUTES).astype(int)

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
        df_mbta = df_mbta.copy()
        for col in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]:
            df_mbta.loc[:, col] = np.nan
        return df_mbta

    df_mbta = df_mbta.copy()
    df_weather = df_weather.copy()
    df_mbta.loc[:, "date"] = df_mbta["date"].astype(str)
    df_weather.loc[:, "date"] = df_weather["date"].astype(str)
    df = pd.merge(df_mbta, df_weather, on="date", how="left")

    weather_cols = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]
    available_weather_cols = [col for col in weather_cols if col in df.columns]

    if available_weather_cols and all(df[col].isna().all() for col in available_weather_cols):
        print("  WARNING: No weather rows matched transit dates; leaving weather columns as NaN.")
        return df

    for col in weather_cols:
        if col in df.columns:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            if pd.notna(median_val) and n_missing > 0:
                print(f"  Imputing {n_missing} missing {col} values with median ({median_val:.1f}).")
                df.loc[:, col] = df[col].fillna(median_val)

    return df


def clean_data(dataset_dir=OFFICIAL_DATASET_DIR, official_standard_type=OFFICIAL_STANDARD_TYPE_DEFAULT):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_massgis_arrival_departure(
        dataset_dir=dataset_dir,
        official_standard_type=official_standard_type,
    )
    df_weather = load_weather()

    df = clean_mbta(df)
    if df.empty:
        print("No data to clean. Exiting.")
        return

    df = add_time_features(df)
    df = merge_weather(df, df_weather)

    final_cols = [
        "route_id", "trip_id", "stop_id", "date", "hour", "day_of_week", "is_weekend", "is_peak",
        "stop_sequence", "scheduled_timestamp", "actual_timestamp",
        "scheduled_arrival", "actual_arrival", "has_actual",
        "delay_minutes", "is_delayed", "is_outlier",
        "direction_id", "time_point_id", "time_point_order", "point_type",
        "standard_type", "scheduled_headway", "headway",
        "TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df_clean = df[final_cols]

    out_path = f"{PROCESSED_DIR}/clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nClean dataset: {len(df_clean)} rows, {df_clean['is_delayed'].mean()*100:.1f}% delayed")
    print(f"Saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean MassGIS MBTA arrival/departure data and merge with weather."
    )
    parser.add_argument(
        "--dataset-dir",
        default=OFFICIAL_DATASET_DIR,
        help=f"Directory of MassGIS MBTA arrival/departure CSVs (default: {OFFICIAL_DATASET_DIR})",
    )
    parser.add_argument(
        "--official-standard-type",
        choices=["schedule", "headway", "all"],
        default=OFFICIAL_STANDARD_TYPE_DEFAULT,
        help=(
            "Which row type to keep: schedule, headway, or all "
            f"(default: {OFFICIAL_STANDARD_TYPE_DEFAULT})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=== Cleaning MassGIS MBTA Data ===\n")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Standard type filter: {args.official_standard_type}")
    clean_data(
        dataset_dir=args.dataset_dir,
        official_standard_type=args.official_standard_type,
    )
    print("\nDone!")
