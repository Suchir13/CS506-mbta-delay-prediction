import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

MBTA_API_KEY = os.getenv("MBTA_API_KEY", "")
BASE_URL = "https://api-v3.mbta.com"
TARGET_ROUTES = ["1", "15", "28", "39", "57"]
DAYS_BACK = 7

def collect_schedules_fallback():
    headers = {"accept": "application/json"}
    if MBTA_API_KEY:
        headers["x-api-key"] = MBTA_API_KEY
    os.makedirs("data/raw", exist_ok=True)
    today = datetime.today()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, DAYS_BACK + 1)]
    all_records = []
    for route_id in TARGET_ROUTES:
        for date_str in dates:
            print(f"  Schedules: route {route_id} on {date_str}...")
            try:
                resp = requests.get(f"{BASE_URL}/schedules",
                    params={"filter[route]": route_id, "filter[date]": date_str,
                            "fields[schedule]": "arrival_time,stop_sequence"},
                    headers=headers, timeout=15)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    attr = item.get("attributes", {})
                    rel = item.get("relationships", {})
                    arr = attr.get("arrival_time")
                    if not arr:
                        continue
                    # Handle both "HH:MM:SS" and "2026-03-23T08:30:00" formats
                    if "T" in str(arr):
                        arr = str(arr).split("T")[1][:8]
                    hour = int(arr.split(":")[0]) % 24
                    delay = np.random.normal(2, 4)
                    if 7 <= hour <= 9 or 16 <= hour <= 19:
                        delay += np.random.normal(3, 2)
                    delay = round(delay, 1)
                    all_records.append({
                        "route_id": route_id,
                        "trip_id": rel.get("trip", {}).get("data", {}).get("id"),
                        "stop_id": rel.get("stop", {}).get("data", {}).get("id"),
                        "date": date_str,
                        "stop_sequence": attr.get("stop_sequence"),
                        "scheduled_arrival": arr,
                        "delay_minutes": delay,
                        "is_delayed": int(delay > 5),
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error: {e}")
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv("data/raw/travel_times.csv", index=False)
        df.to_csv("data/raw/schedules.csv", index=False)
        print(f"\nSaved {len(df)} records to data/raw/travel_times.csv")
        print("NOTE: delays are simulated based on real schedules.")
    else:
        print("No data collected.")

if __name__ == "__main__":
    print("=== Collecting MBTA Bus Data ===")
    collect_schedules_fallback()
    print("\nDone!")
