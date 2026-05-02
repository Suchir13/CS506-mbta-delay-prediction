# Predicting MBTA Bus Delays Using Weather and Time Features
BU CS506 Final Project

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project predicts whether an MBTA bus will arrive more than 5 minutes late using real historical MBTA bus arrival/departure data, Boston weather conditions, and time-based features.

The dataset covers all MBTA bus routes across 15 months from January 2025 through March 2026, enabling network-wide analysis and prediction.

---

## Project Goals

**Primary Goal:**
- Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, route) most strongly influence delays.
- Visualize patterns in delays across routes, times, and weather conditions.
- Provide an interactive predictor that works for any MBTA bus route.

---

## Data Collection

### 1. MBTA Bus Arrival/Departure Data — Official MassGIS / MBTA Dataset
The official MBTA Bus Arrival/Departure dataset published by MassDOT — provides per-stop actual vs scheduled arrival times for all routes back to 2019. No API key required.

**Delay formula:** `delay_minutes = actual_arrival − scheduled_arrival`
**Target variable:** `is_delayed = 1` if delay > 5 minutes

**Coverage:** All MBTA bus routes, January 2025 – March 2026

- 2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
- 2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c

### 2. Boston Weather Data — Open-Meteo
Historical daily weather from [Open-Meteo](https://archive-api.open-meteo.com/v1/archive). No API key required.

- **Location:** Boston (42.3601 N, 71.0589 W)
- **Variables:** Max/min temperature, precipitation, snowfall, wind speed
- Merged with MBTA data by service date

---

## Why MassGIS

The MBTA v3 predictions API is real-time only — once a bus passes a stop, the actual arrival data disappears. Our initial attempts to collect historical predictions consistently returned 400 errors.

The official MassGIS dataset solves this by archiving every per-stop actual vs scheduled arrival time published by MBTA. It is the authoritative source — third-party services like TransitMatters derive their data from this same MBTA feed. We chose MassGIS because:

1. It is the original primary source published directly by MassDOT.
2. It includes richer fields like direction, headway, point type, and stop sequence.
3. It covers every timepoint on every route, not just one segment.
4. Coverage of all MBTA bus routes back to 2019.

---

## Data Cleaning

Implemented in `src/clean_data.py` — no manual edits:

- Drop rows missing route_id, date, or arrival time
- Standardize date format
- Remove duplicate (trip, stop, date) records
- Compute delay: actual − scheduled arrival time
- Flag outliers: |delay| > 120 minutes (kept, marked)
- Impute missing weather values with column median
- Merge weather by service date (no temporal leakage)

---

## Feature Extraction

19 features built in `src/features.py`:

- **Time:** hour of day, day of week, is_weekend, is_peak (rush hour flag)
- **Route/Operational:** route_encoded, direction_encoded, point_type_encoded, standard_type_encoded, stop_sequence, has_actual, scheduled_headway_minutes, scheduled_headway_missing
- **Weather:** is_rainy, is_snowy, TMAX, TMIN, PRCP, SNOW, AWND

---

## Modeling

Three classifiers in `src/train.py`:
1. Logistic Regression — baseline
2. Random Forest — ensemble of decision trees
3. Gradient Boosted Trees — best on validation (F1=0.425)

**Split:** Time-based (no shuffle) — 70% train, 15% validation, 15% test
**Imbalance:** class_weight="balanced" for all models
**Threshold tuning:** Swept 0.10–0.90 on validation, picked best F1 threshold per model
**Hyperparameter sweep:** `src/randomForestParams.py` compares model size vs F1

---

## Evaluation

Full metrics in `src/evaluate.py`:

| Set | F1 | Recall | ROC-AUC |
|-----|----|--------|---------|
| Validation | 0.425 | 0.397 | 0.718 |
| Test | 0.401 | 0.649 | 0.647 |

Per-slice (test set): Peak hours F1=0.651, Off-peak F1=0.398

**Top features:** route_encoded (38%), stop_sequence (17%), hour (15%), TMIN (6%), direction_encoded (5%)

---

## Interactive Predictor

`src/predict.py` lets you enter any MBTA route, hour, day, and weather condition and get a real-time delay prediction from the trained model.

---

## Visualizations

7 plots in `data/processed/plots/`:
- Delay rate by hour of day
- Delay rate by route
- Delay vs precipitation (box plot)
- Confusion matrix
- Feature importance
- PR curve (validation)
- PR curve (test)

---

## Testing

12 unit and integration tests in `tests/test_pipeline.py` — all passing. GitHub Actions runs them on every push.

---

## Timeline

| Week | Goal | Status |
|------|------|--------|
| 1 | API setup, route scoping | ✅ Complete |
| 2 | Real data collection, weather merge | ✅ Complete |
| 3 | Pipeline hardening, EDA | ✅ Complete |
| 4 | Baseline model, March check-in | ✅ Complete |
| 5 | Feature engineering, RF, GBT, hyperparameter tuning | ✅ Complete |
| 6 | Error analysis, feature importance, PR/ROC curves | ✅ Complete |
| 7 | Robustness checks, interactive predictor | ✅ Complete |
| 8 | Final polish, presentation video | ✅ Complete |

---

## Limitations

- Daily weather granularity — hourly would improve signal
- Dataset imbalance means precision is limited — model favours recall
- Computed delay relies on `actual − scheduled` — does not account for cancelled trips or service alerts