# Note: The model is biased towards predicting no delays and mostly says "ON TIME!!"
# As dataset is imbalanced with many more on-time records than delayed ones.
# The model learns to predict the majority class (on time) more often, leading to a bias in predictions.
# To address this, we could try techniques like oversampling the minority class
# (delayed records), undersampling the majority class
import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "data/processed/best_model.pkl"
CLEAN_PATH = "data/processed/clean.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.csv"


def build_mapping(series):
    values = series.astype(str).str.strip()
    values = values.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    categories = sorted(values.dropna().unique().tolist())
    return {value: idx for idx, value in enumerate(categories)}


def load_reference_mappings():
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"{CLEAN_PATH} not found. Run clean_data.py first.")

    ref = pd.read_csv(
        CLEAN_PATH,
        usecols=lambda c: c in {"route_id", "direction_id", "point_type", "standard_type"},
        dtype=str,
        low_memory=False,
    )

    return {
        "route": build_mapping(ref["route_id"]) if "route_id" in ref.columns else {},
        "direction": build_mapping(ref["direction_id"]) if "direction_id" in ref.columns else {},
        "point_type": build_mapping(ref["point_type"]) if "point_type" in ref.columns else {},
        "standard_type": build_mapping(ref["standard_type"]) if "standard_type" in ref.columns else {},
    }


def load_threshold(model_data):
    if "threshold" in model_data:
        return float(model_data["threshold"])

    if os.path.exists(MODEL_RESULTS_PATH):
        try:
            results = pd.read_csv(MODEL_RESULTS_PATH)
            if {"f1", "threshold"}.issubset(results.columns) and len(results) > 0:
                best_idx = results["f1"].astype(float).idxmax()
                return float(results.loc[best_idx, "threshold"])
        except Exception:
            pass

    return 0.5

print("\n=== MBTA Delay Predictor ===\n")

def prompt_with_quit(prompt, default=None):
    raw = input(prompt).strip()
    if raw.lower() == "quit":
        return None
    if raw == "" and default is not None:
        return default
    return raw


def prompt_int(prompt, valid_range=None):
    while True:
        raw = prompt_with_quit(prompt)
        if raw is None:
            return None
        try:
            value = int(raw)
            if valid_range is not None and value not in valid_range:
                print("Invalid value. Please try again or type 'quit' to exit.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer or type 'quit' to exit.")


def build_input_row(features, mappings):
    hour = prompt_int("Enter hour of day (0 to 23): ", valid_range=range(24))
    if hour is None:
        return None

    day_of_week = prompt_int(
        "Enter day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun): ",
        valid_range=range(7),
    )
    if day_of_week is None:
        return None

    routes = mappings["route"]
    print("\nAvailable routes:", ", ".join(routes.keys()))
    route_input = prompt_with_quit("Enter route no: ")
    if route_input is None:
        return None
    while route_input not in routes:
        route_input = prompt_with_quit("Invalid route! Please enter an available one: ")
        if route_input is None:
            return None
    route_encoded = routes[route_input]

    direction_options = list(mappings["direction"].keys())
    direction_default = "Outbound" if "Outbound" in mappings["direction"] else (direction_options[0] if direction_options else "")
    direction_input = prompt_with_quit(
        f"Direction {direction_options} [default={direction_default}]: ",
        default=direction_default,
    )
    if direction_input is None:
        return None

    point_type_options = list(mappings["point_type"].keys())
    point_type_default = "Midpoint" if "Midpoint" in mappings["point_type"] else (point_type_options[0] if point_type_options else "")
    point_type_input = prompt_with_quit(
        f"Point type {point_type_options} [default={point_type_default}]: ",
        default=point_type_default,
    )
    if point_type_input is None:
        return None

    standard_type_default = "Schedule" if "Schedule" in mappings["standard_type"] else (list(mappings["standard_type"].keys())[0] if mappings["standard_type"] else "")

    weather = prompt_with_quit("Weather condition (clear/ rain/ snow) [default=clear]: ", default="clear")
    if weather is None:
        return None
    weather = weather.lower()

    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak = 1 if (day_of_week < 5 and (7 <= hour <= 9 or 16 <= hour <= 19)) else 0
    is_rainy = 1 if weather == "rain" else 0
    is_snowy = 1 if weather == "snow" else 0

    new_data = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_peak": is_peak,
        "route_encoded": route_encoded,
        "direction_encoded": mappings["direction"].get(direction_input, -1),
        "point_type_encoded": mappings["point_type"].get(point_type_input, -1),
        "standard_type_encoded": mappings["standard_type"].get(standard_type_default, -1),
        "stop_sequence": 10,
        "has_actual": 1,
        "scheduled_headway_minutes": 10,
        "scheduled_headway_missing": 0,
        "is_rainy": is_rainy,
        "is_snowy": is_snowy,
        "TMAX": 60,
        "TMIN": 45,
        "PRCP": 0.2 if is_rainy else 0,
        "SNOW": 0.5 if is_snowy else 0,
        "AWND": 5,
    }])

    for feature in features:
        if feature not in new_data.columns:
            new_data[feature] = 0

    return new_data[features], route_input, is_peak, is_rainy, is_snowy


def main():
    # loads model once
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data.get("scaler")
    features = model_data["features"]
    threshold = load_threshold(model_data)
    mappings = load_reference_mappings()

    print("Type 'quit' at any prompt to exit.\n")

    while True:
        built = build_input_row(features, mappings)
        if built is None:
            print("\nExiting predictor.")
            break

        X, route_input, is_peak, is_rainy, is_snowy = built

        try:
            if scaler is not None:
                X = scaler.transform(X)
        except Exception:
            pass

        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        print("\n=== RESULT ===")
        print(f"Delay probability: {round(proba[0], 3)}")

        if pred[0] == 1:
            print("Prediction: Likely DELAY :( (>5 min)")
        else:
            print("Prediction: Likely ON TIME!!")

        print("\nWhy this prediction?")
        if is_peak:
            print("- Peak hours increase delays")
        if is_rainy:
            print("- Rain increases delays")
        if is_snowy:
            print("- Snow increases delays")
        if route_input in ["1", "28"]:
            print("- High-traffic route")

        print("\n--- New query ---\n")
import pandas as pd

#df = pd.read_csv("data/processed/clean.csv")
df = pd.read_csv(
    "data/processed/clean.csv",
    dtype={"route_id": str},
    low_memory=False
)


if __name__ == "__main__":
    main()
