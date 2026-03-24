"""
collect_weather.py
------------------
Downloads historical daily weather data for Boston from the NOAA CDO API.
Saves raw weather data to data/raw/weather.csv.

Weather station: Boston Logan Airport (USW00014739)
Run: python src/collect_weather.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NOAA_TOKEN = os.getenv("NOAA_TOKEN", "")
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# Boston Logan Airport station ID
STATION_ID = "GHCND:USW00014739"

# Match MBTA data range (past 30 days for more weather context)
DAYS_BACK = 30

# Weather data types we want
DATA_TYPES = "TMAX,TMIN,PRCP,SNOW,SNWD,AWND"


def fetch_weather(start_date, end_date):
    """
    Call NOAA CDO API for daily weather summaries.
    start_date / end_date: 'YYYY-MM-DD' strings.
    Returns a DataFrame of weather observations.
    """
    if not NOAA_TOKEN:
        print("WARNING: No NOAA_TOKEN found in .env — skipping weather download.")
        print("Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token")
        return pd.DataFrame()

    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "stationid": STATION_ID,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": DATA_TYPES,
        "limit": 1000,
        "units": "standard",  # Fahrenheit and inches
    }

    print(f"Fetching weather data from {start_date} to {end_date}...")
    try:
        response = requests.get(
            f"{BASE_URL}/data",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        print(f"  Got {len(results)} weather records.")
        return pd.DataFrame(results)
    except requests.RequestException as e:
        print(f"  NOAA API error: {e}")
        return pd.DataFrame()


def reshape_weather(df_raw):
    """
    NOAA returns one row per (date, datatype). Pivot so each date is one row
    with columns: TMAX, TMIN, PRCP, SNOW, SNWD, AWND.
    """
    if df_raw.empty:
        return df_raw

    # Each row has: date, datatype, value
    df_pivot = df_raw.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="mean"
    ).reset_index()

    df_pivot.columns.name = None  # Remove 'datatype' label from column axis
    df_pivot["date"] = pd.to_datetime(df_pivot["date"]).dt.date.astype(str)

    return df_pivot


def collect_weather():
    """Main function to collect and save weather data."""
    os.makedirs("data/raw", exist_ok=True)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_BACK)

    df_raw = fetch_weather(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    if df_raw.empty:
        # Create a placeholder so downstream code doesn't crash
        print("Creating empty weather placeholder...")
        df_placeholder = pd.DataFrame(columns=["date", "TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"])
        df_placeholder.to_csv("data/raw/weather.csv", index=False)
        return

    df_weather = reshape_weather(df_raw)
    df_weather.to_csv("data/raw/weather.csv", index=False)
    print(f"Saved {len(df_weather)} days of weather data to data/raw/weather.csv")


if __name__ == "__main__":
    print("=== Collecting Boston Weather Data ===\n")
    collect_weather()
    print("\nDone!")