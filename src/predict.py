#Note: The model is biased towards predicting no delays and mostly says "ON TIME!!"
# As dataset is imbalanced with many more on-time records than delayed ones.
# The model learns to predict the majority class (on time) more often, leading to a bias in predictions.
# To address this, we could try techniques like oversampling the minority class
# (delayed records), undersampling the majority class
import joblib
import pandas as pd

print('Future Scope Testing...')
print("\n=== MBTA Delay Predictor ===\n")

# loads model
model_data = joblib.load("data/processed/best_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# takes user input

hour = int(input("Enter hour of day (0 to 23): "))
day_of_week = int(input("Enter day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun): "))

# shows available routes
routes = {"1": 0,"15": 1,"28": 2,"39": 3,"57": 4}

print("\nAvailable routes:", ", ".join(routes.keys()))
route_input = input("Enter route no: ")

while route_input not in routes:
    route_input = input("Invalid route! Please, enter an available one: ")

route_encoded = routes[route_input]

# asks user for weather conditions
weather = input("Weather condition (clear/ rain/ snow) [default=clear]: ").lower()


# automatically derives features
is_weekend = 1 if day_of_week >= 5 else 0
is_peak = 1 if (day_of_week < 5 and (7 <= hour <= 9 or 16 <= hour <= 19)) else 0

is_rainy = 1 if weather == "rain" else 0
is_snowy = 1 if weather == "snow" else 0

# default weather values (from Boston weather data averages)
TMAX = 60
TMIN = 45
PRCP = 0.2 if is_rainy else 0
SNOW = 0.5 if is_snowy else 0
AWND = 5

# default placeholders
stop_sequence = 10
has_actual = 1
scheduled_headway_minutes = 10
scheduled_headway_missing = 0


# creates input

new_data = pd.DataFrame([{
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "is_peak": is_peak,
    "route_encoded": route_encoded,
    "stop_sequence": stop_sequence,
    "has_actual": has_actual,
    "scheduled_headway_minutes": scheduled_headway_minutes,
    "scheduled_headway_missing": scheduled_headway_missing,
    "is_rainy": is_rainy,
    "is_snowy": is_snowy,
    "TMAX": TMAX,
    "TMIN": TMIN,
    "PRCP": PRCP,
    "SNOW": SNOW,
    "AWND": AWND
}])

X = new_data[features]

# scales if required
try:
    X = scaler.transform(X)
except:
    pass

# predicts
proba = model.predict_proba(X)[:, 1]
threshold = 0.47
pred = (proba >= threshold).astype(int)


# outputs results
print("\n=== RESULT ===")
print(f"Delay probability: {round(proba[0], 3)}")

if pred[0] == 1:
    print("Prediction: Likely DELAY :( (>5 min)")
else:
    print("Prediction: Likely ON TIME!!")


# explanation of prediction based on input features
print("\nWhy this prediction?")
if is_peak:
    print("- Peak hours increase delays")
if is_rainy:
    print("- Rain increases delays")
if is_snowy:
    print("- Snow increases delays")
if route_input in ["1", "28"]:
    print("- High-traffic route")