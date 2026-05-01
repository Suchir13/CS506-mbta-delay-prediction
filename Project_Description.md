# Predicting MBTA Bus Delays Using Weather and Time Features
BU CS506 Final Project

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project predicts whether an MBTA bus will arrive more than 5 minutes late using real historical bus travel-time data, Boston weather conditions, and time-based features.

Understanding delay patterns helps identify factors contributing to unreliable service and provides insights for transit optimization.

---

## Project Goals

**Primary Goal:**
- Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, route) most strongly influence delays.
- Visualize patterns in delays across routes, times, and weather conditions.
- Provide an interactive predictor so users can query the trained model directly.

---

## Data Collection

### 1. MBTA Travel Time Data — TransitMatters API
Real historical bus travel-time data is collected from the [TransitMatters Dashboard API](https://dashboard-api.labs.transitmatters.org) — a free, publicly available archive of MBTA GTFS-RT data. No API key required.

The MBTA v3 predictions endpoint is real-time only — once a bus passes a stop, the data is gone. TransitMatters archives this historically, making it the only free source of real past delays.

**Delay formula:** `delay_minutes = (actual_travel_time − benchmark_travel_time) / 60`
**Target variable:** `is_delayed = 1` if delay > 5 minutes

**Routes:** 1 (Harvard→Nubian), 15 (Uphams Corner→Ruggles), 28 (Mattapan→Nubian), 39 (Forest Hills→Back Bay), 57 (Watertown→Kenmore)

**Dataset:** 131,753 real trip records, January 2025 – October 2025, 17.9% delayed

### 2. Boston Weather Data — Open-Meteo
Historical daily weather from [Open-Meteo](https://archive-api.open-meteo.com/v1/archive). No API key required.

- **Location:** Boston (42.3601 N, 71.0589 W)
- **Variables:** Max/min temperature, precipitation, snowfall, wind speed
- **Dataset:** 304 days, January 2025 – October 2025
- Merged with MBTA data by service date

### 3. Official MassGIS / MBTA Dataset (also supported)
The official MBTA Bus Arrival/Departure dataset published by MassDOT provides per-stop actual vs scheduled arrival times for all routes back to 2019 — 35+ million records. The pipeline supports this via `--source official`.

- 2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
- 2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c

---

## Data Cleaning

Implemented in `src/clean_data.py` — no manual edits:

- Drop rows missing route_id, date, or arrival time
- Standardize date format
- Remove duplicate (trip, stop, date) records
- Compute delay: actual − benchmark travel time
- Flag outliers: |delay| > 120 minutes (kept, marked)
- Impute missing weather values with column median
- Merge weather by service date (no temporal leakage)

---

## Feature Extraction

15 features built in `src/features.py`:

- **Time:** hour of day, day of week, is_weekend, is_peak (rush hour flag)
- **Route:** route_encoded (numeric), has_actual, scheduled_headway_minutes, scheduled_headway_missing
- **Weather:** is_rainy, is_snowy, TMAX, TMIN, PRCP, SNOW, AWND

---

## Modeling

Three classifiers in `src/train.py`:
1. Logistic Regression — baseline
2. Random Forest — best on validation (F1=0.422)
3. Gradient Boosted Trees

**Split:** Time-based (no shuffle) — 70% train, 15% validation, 15% test
**Imbalance:** class_weight="balanced" for all models
**Threshold tuning:** Swept 0.10–0.90 on validation, picked best F1 threshold per model
**Hyperparameter sweep:** `src/randomForestParams.py` compares model size vs F1

---

## Evaluation

Full metrics in `src/evaluate.py`:

| Set | F1 | Recall | ROC-AUC | PR-AUC |
|-----|----|--------|---------|--------|
| Validation | 0.422 | 0.864 | 0.596 | 0.332 |
| Test | 0.254 | 0.765 | 0.544 | 0.157 |

Per-slice (test set): Peak hours F1=0.314, Off-peak F1=0.212

**Top features:** route_encoded (58.7%), hour (12.5%), TMAX (7.2%), TMIN (6.9%), AWND (4.3%)

---

## Interactive Predictor

`src/predict.py` lets you enter a route, hour, day, and weather condition and get a real-time delay prediction from the trained model.

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

- One stop-pair per route (inbound main segment only)
- Daily weather granularity — hourly would improve signal
- Val→test F1 gap reflects seasonal threshold drift
- Dataset imbalance (82% on-time) limits precision
- MassGIS official data would significantly improve coverage