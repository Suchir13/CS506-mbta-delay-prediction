"""
collect_weather.py
------------------
Downloads historical daily weather data for Boston from Open-Meteo.
No API key required.

Weather location: Boston (42.3601 N, 71.0589 W)
Run: python src/collect_weather.py
"""

import os
import requests
import pandas as pd

LATITUDE  = 42.3601
LONGITUDE = -71.0589
OUTPUT    = "data/raw/weather.csv"

# Match the travel time data range collected from TransitMatters
START_DATE = "2024-01-01"
END_DATE   = "2025-03-31"


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching Boston weather from Open-Meteo ({start_date} to {end_date})...")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":           LATITUDE,
            "longitude":          LONGITUDE,
            "start_date":         start_date,
            "end_date":           end_date,
            "daily":              ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "snowfall_sum",
                "wind_speed_10m_max",
            ]),
            "temperature_unit":   "fahrenheit",
            "precipitation_unit": "inch",
            "wind_speed_unit":    "mph",
            "timezone":           "America/New_York",
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "TMAX": daily["temperature_2m_max"],
        "TMIN": daily["temperature_2m_min"],
        "PRCP": daily["precipitation_sum"],
        "SNOW": daily["snowfall_sum"],
        "AWND": daily["wind_speed_10m_max"],
    })
    print(f"  Got {len(df)} days of weather data.")
    return df


def collect_weather(start: str = START_DATE, end: str = END_DATE, append: bool = False):
    os.makedirs("data/raw", exist_ok=True)
    df_new = fetch_weather(start, end)

    if append and os.path.exists(OUTPUT):
        df_existing = pd.read_csv(OUTPUT)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["date"], inplace=True)
        df_combined.sort_values("date", inplace=True)
        df_combined.to_csv(OUTPUT, index=False)
        print(f"Appended -> {OUTPUT}  (total: {len(df_combined)} days)")
        print(f"Full date range: {df_combined['date'].min()} -> {df_combined['date'].max()}")
    else:
        df_new.to_csv(OUTPUT, index=False)
        print(f"Saved {len(df_new)} days to {OUTPUT}")
        print(f"Date range: {df_new['date'].min()} -> {df_new['date'].max()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default=START_DATE)
    parser.add_argument("--end",    default=END_DATE)
    parser.add_argument("--append", action="store_true",
                        help="Append to existing weather.csv instead of overwriting")
    args = parser.parse_args()

    print("=== Collecting Boston Weather Data (Open-Meteo) ===\n")
    collect_weather(args.start, args.end, args.append)
    print("\nDone!")
