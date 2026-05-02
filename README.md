# Predicting MBTA Bus Delays Using Weather and Time Features
**BU CS 506 Final Project**

> 🎥 **Presentation Video:** https://www.youtube.com/watch?v=X4RSpT5v3qY

---

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project predicts whether an MBTA bus will arrive **more than 5 minutes late** using real historical MBTA bus arrival/departure data combined with Boston weather conditions and time-based features.

The dataset covers all MBTA bus routes across 15 months (Jan 2025 – Mar 2026), enabling network-wide analysis and prediction.

---

## How to Build and Run

### 1. Prerequisites
- Python 3.9 or newer
- `pip`

### 2. Clone and Install
```bash
git clone https://github.com/Suchir13/CS506-mbta-delay-prediction.git
cd CS506-mbta-delay-prediction
pip install -r requirements.txt
```

### 3. Reproduce Results (No Downloads Needed)
All cleaned data and the trained model are already included. Run these to reproduce every result and regenerate all plots:

```bash
python src/features.py
python src/train.py
python src/visualize.py
python src/evaluate.py
```

### 4. Run Tests
```bash
pytest tests/ -v
```
All 12 tests should pass.

### 5. Try the Interactive Predictor
Enter any MBTA route, hour, day, and weather condition to get a live delay prediction:
```bash
python src/predict.py
```

---

## (Optional) Re-run From Raw Data

The processed data is already committed, so you do not need to do this. Only follow these steps if you want to re-collect fresh data.

### Download fresh MBTA data
Download monthly CSV files from the official MassGIS / MBTA dataset — no API key required:
- **2025 data:** https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
- **2026 data:** https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c

Place all downloaded CSV files in `data/raw/arrival_departure/`.

### Re-collect weather to match the date range
```bash
python src/collect_weather.py --start 2025-01-01 --end 2026-03-31
```

### Re-run the full pipeline from raw data
```bash
python src/clean_data.py
python src/features.py
python src/train.py
python src/visualize.py
python src/evaluate.py
```

---

## Project Goals

**Primary Goal:** Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, route) most strongly influence delays.
- Visualize delay patterns across routes, times, and weather conditions.
- Provide an interactive predictor that works for any MBTA route.

---

## Data Collection

### Source 1 — MBTA Bus Arrival/Departure Data (Official MassGIS / MBTA)
The official MBTA Bus Arrival/Departure dataset published by MassDOT — provides per-stop actual vs scheduled arrival times for all routes back to 2019. No API key required.

- **2025 dataset:** https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
- **2026 dataset:** https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c
- 35+ million records with `scheduled` and `actual` timestamps per stop
- Includes direction (Inbound/Outbound), headway, point type, and stop sequence
- **Delay formula:** `delay_minutes = actual_arrival − scheduled_arrival`
- **Target:** `is_delayed = 1` if delay > 5 minutes, else `0`

**Coverage:** All MBTA bus routes, January 2025 – March 2026

### Source 2 — Boston Weather: Open-Meteo
Historical daily weather from [Open-Meteo](https://archive-api.open-meteo.com/v1/archive). No API key required.

- **Location:** Boston (42.3601 N, 71.0589 W)
- **Variables:** TMAX, TMIN, PRCP (precipitation), SNOW, AWND (wind speed)
- **Units:** °F, inches, mph
- Merged with MBTA data by service date
- **Dataset:** January 2025 – March 2026

---

## Data Cleaning

All cleaning logic is in `src/clean_data.py` — no manual edits anywhere.

| Step | Action |
|------|--------|
| Drop missing keys | Remove rows missing `route_id`, `date`, or arrival time |
| Normalize fields | Standardize date format, strip whitespace |
| Deduplication | Keep first occurrence per (trip, stop, date) |
| Delay computation | `delay_minutes = actual_arrival − scheduled_arrival` |
| Outlier flagging | Flag \|delay\| > 120 min — kept but marked `is_outlier=1` |
| Weather imputation | Fill missing weather values with column median |
| Weather merge | Join on service date — no future data leakage |

---

## Feature Extraction

Features built in `src/features.py` (19 total):

| Feature | Description |
|---------|-------------|
| `hour` | Hour of scheduled arrival (0–23) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_peak` | 1 if weekday 7–9 AM or 4–7 PM |
| `route_encoded` | Numeric encoding of route ID |
| `direction_encoded` | Encoded direction (Inbound/Outbound) |
| `point_type_encoded` | Encoded stop type (Startpoint/Midpoint/Endpoint) |
| `standard_type_encoded` | Encoded MBTA standard classification of stop |
| `stop_sequence` | Position of the stop along the route |
| `has_actual` | 1 if actual arrival time was recorded |
| `scheduled_headway_minutes` | Planned gap between buses (minutes) |
| `scheduled_headway_missing` | 1 if headway data was absent |
| `is_rainy` | 1 if precipitation > 0.1 inches |
| `is_snowy` | 1 if snowfall > 0.1 inches |
| `TMAX` / `TMIN` | Daily high/low temperature (°F) |
| `PRCP` | Daily precipitation (inches) |
| `SNOW` | Daily snowfall (inches) |
| `AWND` | Average wind speed (mph) |

> `is_delayed` is our target variable (1 if delay > 5 minutes, else 0)

> `route_avg_delay` was excluded — computing it before the train/test split introduces data leakage.

---

## Modeling

Three classifiers trained and compared in `src/train.py`:

1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — ensemble of decision trees, captures non-linear patterns
3. **Gradient Boosted Trees** — sequential boosting, typically strongest

**Split strategy (time-based, no shuffle):**
- Train: first 70% chronologically
- Validation: next 15%
- Test: final 15%

**Class imbalance handling:**
- Logistic Regression and Random Forest: `class_weight="balanced"`
- Gradient Boosted Trees: `compute_sample_weight("balanced")` at fit time

**Threshold tuning:** Thresholds 0.10–0.90 swept on validation set; threshold maximising F1 selected per model.

**Hyperparameter tuning:** `src/randomForestParams.py` sweeps multiple Random Forest configurations.

---

## Results

### Validation Set

| Model | F1 | Threshold |
|-------|----|-----------|
| Logistic Regression (baseline) | 0.331 | 0.56 |
| Random Forest | 0.399 | 0.60 |
| **Gradient Boosted Trees (best)** | **0.425** | **0.67** |

### Test Set (best model: Gradient Boosted Trees)

| Metric | Score |
|--------|-------|
| Accuracy | 0.553 |
| Precision | 0.290 |
| Recall | 0.649 |
| F1 | 0.401 |
| ROC-AUC | 0.647 |

### Per-Slice Performance (Test Set, from `evaluate.py`)

| Slice | Accuracy | F1 |
|-------|----------|----|
| Peak hours | 0.596 | 0.651 |
| Off-peak hours | 0.552 | 0.398 |

**Top Feature Importances (Gradient Boosted Trees):**

| Feature | Importance |
|---------|-----------|
| `route_encoded` | 38.0% |
| `stop_sequence` | 16.9% |
| `hour` | 15.3% |
| `TMIN` | 5.9% |
| `direction_encoded` | 4.5% |
| `AWND` | 3.8% |

**Key Findings:**
- `route_encoded` is the strongest predictor (38%) — different routes have genuinely different delay rates
- Stop sequence and hour together account for ~32% — where on the route and what time both matter
- Weather features (TMIN, AWND, TMAX) contribute meaningfully
- High Recall (0.649) means the model catches ~65% of real delays — useful for commuter warning systems

---

## Visualizations

All plots saved to `data/processed/plots/`.

| Plot | File | How to generate |
|------|------|----------------|
| Delay rate by hour | `delay_by_hour.png` | `python src/visualize.py` |
| Delay rate by route | `delay_by_route.png` | `python src/visualize.py` |
| Delay vs precipitation | `delay_vs_precip.png` | `python src/visualize.py` |
| Confusion matrix | `confusion_matrix.png` | `python src/visualize.py` |
| Feature importances | `feature_importance.png` | `python src/visualize.py` |
| PR curve (validation) | `pr_curve_validation.png` | `python src/evaluate.py` |
| PR curve (test) | `pr_curve_test.png` | `python src/evaluate.py` |

---

## Interactive Predictor

`src/predict.py` loads the trained model and lets you query it interactively for any MBTA bus route in the dataset:

```
$ python src/predict.py

=== MBTA Delay Predictor ===

Type 'quit' at any prompt to exit.

Enter hour of day (0 to 23): 8
Enter day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun): 2

Available routes: <full list of MBTA routes>
Enter route no: 28
Direction ['Inbound', 'Outbound'] [default=Outbound]: Inbound
Point type ['Endpoint', 'Midpoint', 'Startpoint'] [default=Midpoint]: Midpoint
Weather condition (clear/ rain/ snow) [default=clear]: snow

=== RESULT ===
Delay probability: 0.412
Prediction: Likely DELAYED!

Why this prediction?
- Peak hours increase delays
- Snow increases delays
- High-traffic route
```

> Note: The model is biased towards on-time because the dataset is imbalanced. High Recall means real delays are still caught ~65% of the time.

---

## Testing

12 unit and integration tests in `tests/test_pipeline.py` — all passing:
- Time string parsing including MBTA's >24-hour format
- Hour, weekday, and peak-hour feature extraction
- Rain and snow flag thresholds
- Route encoding consistency
- End-to-end delay label correctness

```bash
pytest tests/ -v
```

GitHub Actions runs these automatically on every push via `.github/workflows/tests.yml`.

---

## Repository Structure

```
CS506-mbta-delay-prediction/
├── src/
│   ├── collect_weather.py          # Download Boston weather from Open-Meteo
│   ├── clean_data.py               # Merge, compute delays, clean MassGIS data
│   ├── features.py                 # Feature engineering (19 features)
│   ├── train.py                    # Train 3 models, pick best
│   ├── evaluate.py                 # PR/ROC curves, peak/route slicing
│   ├── randomForestParams.py       # Random Forest hyperparameter sweep
│   ├── predict.py                  # Interactive delay predictor
│   └── visualize.py                # Generate 5 EDA/results plots
├── tests/
│   └── test_pipeline.py            # 12 unit + integration tests
├── data/
│   ├── raw/
│   │   └── weather.csv             # Daily Boston weather (Jan 2025–Mar 2026)
│   └── processed/
│       ├── clean.csv               # All MBTA routes — cleaned and merged
│       ├── features.csv            # Final feature matrix (19 features)
│       ├── model_results.csv       # Validation F1 per model
│       ├── best_model.pkl          # Saved best model + scaler
│       ├── split_info.csv          # Train/val/test split indices
│       ├── val_predictions.csv     # Validation set predictions
│       ├── test_predictions.csv    # Test set predictions
│       └── plots/                  # 7 generated visualizations
├── .github/workflows/
│   └── tests.yml                   # CI: run tests on every push
├── Project_Description.md
├── requirements.txt
└── README.md
```

---

## Progress Against Project Timeline

| Week | Goal | Status |
|------|------|--------|
| 1 | API setup, route scoping | ✅ Complete |
| 2 | Real data collection, weather merge | ✅ Complete — MassGIS + Open-Meteo |
| 3 | Pipeline hardening, EDA, outlier handling | ✅ Complete |
| 4 | Baseline Logistic Regression, March check-in | ✅ Complete |
| 5 | Feature engineering, Random Forest, GBT, hyperparameter tuning | ✅ Complete |
| 6 | Error analysis, feature importance, PR/ROC curves | ✅ Complete |
| 7 | Robustness checks (peak/off-peak, per-route), interactive predictor | ✅ Complete |
| 8 | Final polish, 10-min presentation video | ✅ Complete |

---

## Environment

- Python 3.9+
- macOS, Linux, or Windows (WSL recommended on Windows)
- All dependencies in `requirements.txt`

---

## Limitations

- Weather merged at daily granularity — hourly weather would improve signal
- Dataset imbalance means precision is limited — model favours recall to catch more real delays
- Delays simulated by `actual − scheduled` from MassGIS — does not account for cancelled trips or service alerts