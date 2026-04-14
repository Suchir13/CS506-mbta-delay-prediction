"""
tests/test_pipeline.py
-----------------------
Unit tests for the data cleaning and feature engineering pipeline.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_data import parse_time_to_seconds, add_time_features, clean_mbta
from features import add_rain_snow_flags, encode_route


# ─── Tests for parse_time_to_seconds ──────────────────────────────────────────

def test_parse_normal_time():
    """Standard HH:MM:SS should convert correctly."""
    assert parse_time_to_seconds("08:30:00") == 8 * 3600 + 30 * 60

def test_parse_midnight_time():
    """25:15:00 is MBTA's way of writing 1:15 AM the next day."""
    assert parse_time_to_seconds("25:15:00") == 25 * 3600 + 15 * 60

def test_parse_missing_time():
    """NaN input should return NaN."""
    result = parse_time_to_seconds(np.nan)
    assert np.isnan(result)

def test_parse_empty_string():
    """Empty string input should return NaN."""
    result = parse_time_to_seconds("")
    assert np.isnan(result)


# ─── Tests for add_time_features ──────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A small DataFrame mimicking clean MBTA data."""
    return pd.DataFrame({
        "route_id": ["1", "1", "28", "28"],
        "trip_id": ["t1", "t2", "t3", "t4"],
        "stop_id": ["s1", "s2", "s3", "s4"],
        "date": ["2024-03-04", "2024-03-04", "2024-03-09", "2024-03-09"],  # Mon, Sat
        "scheduled_arrival": ["08:30:00", "17:15:00", "10:00:00", "22:00:00"],
        "delay_minutes": [3.0, 8.0, -1.0, 6.0],
        "is_delayed": [0, 1, 0, 1],
        "is_outlier": [False, False, False, False],
    })

def test_hour_extraction(sample_df):
    """Hour should be correctly extracted from scheduled_arrival."""
    df = add_time_features(sample_df.copy())
    assert df.loc[0, "hour"] == 8
    assert df.loc[1, "hour"] == 17

def test_weekend_flag(sample_df):
    """March 9 2024 is a Saturday — should be marked as weekend."""
    df = add_time_features(sample_df.copy())
    assert df.loc[0, "is_weekend"] == 0  # Monday
    assert df.loc[2, "is_weekend"] == 1  # Saturday

def test_peak_hour_flag(sample_df):
    """Morning rush (8 AM weekday) should be peak; Saturday should not."""
    df = add_time_features(sample_df.copy())
    assert df.loc[0, "is_peak"] == 1   # 8 AM Monday — peak
    assert df.loc[2, "is_peak"] == 0   # 10 AM Saturday — not peak (weekend)


# ─── Tests for add_rain_snow_flags ────────────────────────────────────────────

def test_rain_flag():
    df = pd.DataFrame({"PRCP": [0.0, 0.05, 0.2, 1.0]})
    result = add_rain_snow_flags(df)
    expected = [0, 0, 1, 1]
    assert list(result["is_rainy"]) == expected

def test_snow_flag():
    df = pd.DataFrame({"SNOW": [0.0, 0.0, 0.5, 3.0]})
    result = add_rain_snow_flags(df)
    expected = [0, 0, 1, 1]
    assert list(result["is_snowy"]) == expected

def test_no_weather_columns():
    """If weather columns are missing, flags should default to 0."""
    df = pd.DataFrame({"delay_minutes": [1, 2]})
    result = add_rain_snow_flags(df)
    assert list(result["is_rainy"]) == [0, 0]
    assert list(result["is_snowy"]) == [0, 0]


# ─── Tests for encode_route ───────────────────────────────────────────────────

def test_encode_route():
    df = pd.DataFrame({"route_id": ["39", "1", "28", "1", "39"]})
    result = encode_route(df)
    # Encoding should be consistent (same route → same code)
    assert result.loc[0, "route_encoded"] == result.loc[4, "route_encoded"]
    assert result.loc[1, "route_encoded"] == result.loc[3, "route_encoded"]
    # Different routes → different codes
    assert result.loc[0, "route_encoded"] != result.loc[1, "route_encoded"]


# ─── Integration: clean_mbta pipeline ────────────────────────────────────────

def test_clean_mbta_delay_and_target():
    """clean_mbta should preserve delay_minutes and set is_delayed correctly."""
    df = pd.DataFrame({
        "route_id": ["1", "28"],
        "trip_id": ["t1", "t2"],
        "stop_id": ["s1", "s2"],
        "date": ["2024-03-04", "2024-03-09"],
        "stop_sequence": [1, 1],
        "scheduled_arrival": ["08:00:00", "10:00:00"],
        "delay_minutes": [7.0, 2.0],   # 7 min late → delayed; 2 min → not delayed
        "is_delayed": [1, 0],
    })
    result = clean_mbta(df)
    assert len(result) == 2
    assert result.loc[result["delay_minutes"] == 7.0, "is_delayed"].values[0] == 1
    assert result.loc[result["delay_minutes"] == 2.0, "is_delayed"].values[0] == 0