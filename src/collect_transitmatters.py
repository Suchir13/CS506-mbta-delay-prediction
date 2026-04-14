"""
Collect real historical MBTA bus delay data from TransitMatters.

Source: TransitMatters public API (dashboard-api.labs.transitmatters.org)
Data: trip-level travel times vs benchmark (scheduled) times, 2024-01-01 onward.

Outputs: data/raw/travel_times.csv  (replaces simulated data, same schema as before)

Usage:
    python src/collect_transitmatters.py
    python src/collect_transitmatters.py --start 2024-01-01 --end 2024-12-31
"""

import os, sys, time, argparse, requests, pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://dashboard-api.labs.transitmatters.org"

# Route -> (from_stop, to_stop) for the main inbound segment.
# Stop IDs confirmed from TransitMatters /api/stops/{route}.
ROUTE_STOPS = {
    "1":  ("1-1-110",   "1-1-64"),      # Harvard -> Nubian
    "15": ("15-1-1480", "15-1-17861"),  # Uphams Corner -> Ruggles
    "28": ("28-1-18511","28-1-64000"),  # Mattapan -> Nubian
    "39": ("39-1-10642","39-1-23391"),  # Forest Hills -> Back Bay
    "57": ("57-1-900",  "57-1-899"),    # Watertown -> Kenmore
}

OUTPUT_FILE = "data/raw/travel_times.csv"
SCHEMA = ["route_id","trip_id","stop_id","date",
          "stop_sequence","scheduled_arrival","delay_minutes","is_delayed"]


def date_range(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while d <= stop:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def fetch_day(route_id: str, from_stop: str, to_stop: str, date_str: str):
    """
    Fetch trip-level travel times for one route/segment on one date.
    Returns a list of row dicts matching the pipeline schema.
    """
    url = f"{BASE_URL}/api/traveltimes/{date_str}"
    try:
        resp = requests.get(url, params={"from_stop": from_stop, "to_stop": to_stop},
                            timeout=20)
        resp.raise_for_status()
        trips = resp.json()
    except Exception as e:
        print(f"    [{route_id} {date_str}] fetch error: {e}")
        return []

    rows = []
    for i, t in enumerate(trips):
        benchmark = t.get("benchmark_travel_time_sec")
        actual    = t.get("travel_time_sec")
        dep_dt    = t.get("dep_dt", "")

        # Skip if we can't compute a real delay
        if benchmark is None or actual is None or not dep_dt:
            continue
        if benchmark <= 0:
            continue

        delay_min = round((actual - benchmark) / 60.0, 2)

        # Parse departure datetime
        try:
            dep = datetime.fromisoformat(dep_dt)
        except ValueError:
            continue

        vehicle = t.get("vehicle_label") or f"{route_id}-{i}"

        rows.append({
            "route_id":         route_id,
            "trip_id":          vehicle,
            "stop_id":          from_stop,
            "date":             dep.strftime("%Y-%m-%d"),
            "stop_sequence":    1,
            "scheduled_arrival": dep.strftime("%H:%M:%S"),
            "delay_minutes":    delay_min,
            "is_delayed":       int(delay_min > 5),
        })

    return rows


def collect(start: str, end: str):
    os.makedirs("data/raw", exist_ok=True)
    dates = list(date_range(start, end))
    total_dates = len(dates)

    all_rows = []
    for route_id, (from_stop, to_stop) in ROUTE_STOPS.items():
        print(f"\nRoute {route_id}  ({from_stop} -> {to_stop})  "
              f"{start} to {end}  [{total_dates} days]")
        for idx, d in enumerate(dates, 1):
            rows = fetch_day(route_id, from_stop, to_stop, d)
            all_rows.extend(rows)
            if idx % 30 == 0 or idx == total_dates:
                print(f"  {idx}/{total_dates} days  ({len(all_rows)} rows so far)")
            time.sleep(0.25)   # be polite to the API

    if not all_rows:
        print("No data collected — check stop IDs or date range.")
        return

    df = pd.DataFrame(all_rows, columns=SCHEMA)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} real trip records to {OUTPUT_FILE}")
    print(f"Date range in data: {df['date'].min()} -> {df['date'].max()}")
    print(f"Delayed trips: {df['is_delayed'].mean():.1%}")
    print(f"Avg delay: {df['delay_minutes'].mean():.1f} min")


def collect_append(start: str, end: str):
    """Collect new date range and append to existing OUTPUT_FILE."""
    new_rows = []
    dates = list(date_range(start, end))
    total_dates = len(dates)

    for route_id, (from_stop, to_stop) in ROUTE_STOPS.items():
        print(f"\nRoute {route_id}  ({from_stop} -> {to_stop})  "
              f"{start} to {end}  [{total_dates} days]")
        for idx, d in enumerate(dates, 1):
            rows = fetch_day(route_id, from_stop, to_stop, d)
            new_rows.extend(rows)
            if idx % 30 == 0 or idx == total_dates:
                print(f"  {idx}/{total_dates} days  ({len(new_rows)} new rows so far)")
            time.sleep(0.25)

    if not new_rows:
        print("No new data collected.")
        return

    df_new = pd.DataFrame(new_rows, columns=SCHEMA)

    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE, low_memory=False)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(
            subset=["route_id", "trip_id", "stop_id", "date", "scheduled_arrival"],
            inplace=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAppended {len(df_new)} rows -> {OUTPUT_FILE} "
              f"(total: {len(df_combined)} rows)")
        print(f"Full date range: {df_combined['date'].min()} -> {df_combined['date'].max()}")
    else:
        df_new.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(df_new)} rows -> {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01",
                        help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end",   default="2025-03-31",
                        help="End date YYYY-MM-DD (default: 2025-03-31)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing data instead of overwriting")
    args = parser.parse_args()

    print(f"=== Collecting TransitMatters historical bus data ===")
    print(f"Period: {args.start} to {args.end}")
    if args.append:
        collect_append(args.start, args.end)
    else:
        collect(args.start, args.end)
    print("\nDone! Now run:")
    print("  python src/clean_data.py")
    print("  python src/features.py")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
