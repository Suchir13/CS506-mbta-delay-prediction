# Predicting MBTA Bus Delays Using Weather and Time Features
**BU CS 506 Final Project**

> 🎥 **Presentation Video:** [Link to be added after recording]

---

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project predicts whether an MBTA bus will arrive **more than 5 minutes late** using real historical bus travel-time data combined with Boston weather conditions and time-based features.

Understanding delay patterns helps identify factors contributing to unreliable service and provides insights for transit optimization.

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

### 3. Run the Full Pipeline
Data is already included in the repo — no API keys needed to reproduce results.
```bash
python src/clean_data.py
python src/features.py
python src/train.py
python src/visualize.py
```

### 4. Run Tests
```bash
pytest tests/ -v
```

### 5. (Optional) Deep Evaluation
Generates additional metrics, ROC/PR curves, and per-route/per-hour performance slices:
```bash
python src/evaluate.py
```

### 6. (Optional) Re-collect fresh data
Only needed if you want to download new data — **no API keys required**:
```bash
# Re-collect MBTA travel times from TransitMatters (free, no auth)
python src/collect_transitmatters.py --start 2025-01-01 --end 2025-10-31

# Re-collect Boston weather from Open-Meteo (free, no auth)
python src/collect_weather.py --start 2025-01-01 --end 2025-10-31
```

---

## Project Goals

**Primary Goal:** Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, day of week) most strongly influence delays.
- Visualize delay patterns across routes, times, and weather conditions.

---

## Data Collection

### MBTA Travel Time Data — TransitMatters API
Real historical bus travel-time data is collected from the [TransitMatters Dashboard API](https://dashboard-api.labs.transitmatters.org) — a free, publicly available archive of MBTA GTFS-RT data. No API key is required.

| Field | What it provides |
|-------|-----------------|
| `travel_time_sec` | Actual travel time for each trip (seconds) |
| `benchmark_travel_time_sec` | Scheduled/expected travel time (seconds) |
| `dep_dt` | Departure datetime |

**Delay formula:** `delay_minutes = (actual − benchmark) / 60`  
**Target variable:** `is_delayed = 1` if `delay_minutes > 5`, else `0`

**Target routes:** 1, 15, 28, 39, 57 (high-traffic Boston corridors)

| Route | Segment |
|-------|---------|
| 1 | Harvard → Nubian |
| 15 | Uphams Corner → Ruggles |
| 28 | Mattapan → Nubian |
| 39 | Forest Hills → Back Bay |
| 57 | Watertown → Kenmore |

**Dataset:** 131,753 trip records, January 2025 – October 2025

> **Why TransitMatters instead of the MBTA API?**  
> The MBTA v3 `/predictions` endpoint is real-time only — once a bus passes a stop, the prediction is gone. TransitMatters archives MBTA GTFS-RT data and exposes it via a free historical API.

### Boston Weather Data — Open-Meteo
Historical daily weather is collected from [Open-Meteo](https://archive-api.open-meteo.com/v1/archive). No API key required.

- **Location:** Boston (42.3601 N, 71.0589 W)
- **Variables:** TMAX, TMIN, PRCP (precipitation), SNOW, AWND (wind speed)
- **Units:** °F, inches, mph
- Merged with MBTA data by service date.

---

## Data Cleaning

All cleaning logic is in `src/clean_data.py` and applied programmatically (no manual edits).

| Step | Action |
|------|--------|
| Drop missing keys | Remove rows missing `route_id`, `trip_id`, `stop_id`, or arrival time |
| Deduplication | Keep first occurrence per (trip, stop, date) |
| Time parsing | Handle MBTA's 25:xx:xx format (trips past midnight) |
| Outlier flagging | Flag rows with \|delay\| > 120 min (kept but marked) |
| Weather merge | Join on service date — no future data leakage |

---

## Feature Extraction

Features used in the model (12 total):

| Feature | Description |
|---------|-------------|
| `hour` | Hour of scheduled arrival (0–23) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_peak` | 1 if weekday 7–9 AM or 4–7 PM |
| `route_encoded` | Numeric encoding of route ID |
| `is_rainy` | 1 if precipitation > 0.1 inches |
| `is_snowy` | 1 if snowfall > 0.1 inches |
| `TMAX` / `TMIN` | Daily high/low temperature (°F) |
| `PRCP` | Daily precipitation (inches) |
| `SNOW` | Daily snowfall (inches) |
| `AWND` | Average wind speed (mph) |

> **Note:** `route_avg_delay` was excluded from the model — computing it on the full dataset before splitting introduces data leakage (the feature "knows" about the test set).

---

## Modeling

Three classifiers are trained and compared in `src/train.py`:

1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — captures non-linear interactions
3. **Gradient Boosted Trees** — typically best performance

**Split strategy (time-based, no shuffle — preserves chronological order to avoid temporal leakage):**
- Train: first 70% of data
- Validation: next 15%
- Test: final 15%

**Class imbalance handling:**
- Logistic Regression and Random Forest: `class_weight="balanced"`
- Gradient Boosted Trees: `compute_sample_weight("balanced")` passed at fit time

**Threshold tuning:** For each model, thresholds from 0.10 to 0.90 are swept on the validation set and the threshold maximising F1 is selected.

**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

The best model (by validation F1) is saved to `data/processed/best_model.pkl`.

---

## Results

### Validation Set

| Model | Val F1 | Threshold |
|-------|--------|-----------|
| Logistic Regression (baseline) | 0.405 | 0.42 |
| Random Forest | 0.416 | 0.23 |
| **Gradient Boosted Trees (best)** | **0.427** | **0.64** |

### Test Set (best model: Gradient Boosted Trees)

| Metric | Score |
|--------|-------|
| F1 | 0.215 |
| ROC-AUC | 0.550 |

**Key Findings:**
- Time-of-day features (`hour`, `is_peak`) are the strongest delay predictors
- Peak hours (7–9 AM, 4–7 PM weekdays) show significantly higher delay rates
- Weather features provide marginal additional signal
- The validation–test F1 gap (~0.21) reflects seasonal threshold shift — the model was tuned on a summer/fall validation window and tested on a later fall window with a different delay distribution

---

## Visualizations

All plots are saved to `data/processed/plots/` after running `python src/visualize.py`.

| Plot | File |
|------|------|
| Delay rate by hour | `delay_by_hour.png` |
| Delay rate by route | `delay_by_route.png` |
| Delay vs precipitation | `delay_vs_precip.png` |
| Confusion matrix | `confusion_matrix.png` |
| Feature importances | `feature_importance.png` |

---

## Testing

Tests cover the core data processing logic in `tests/test_pipeline.py`:
- Time string parsing (including MBTA's >24-hour format)
- Hour / weekday / peak-hour feature extraction
- Rain and snow flag generation
- Route encoding consistency
- End-to-end delay label correctness

```bash
pytest tests/ -v
```

---

## Repository Structure

```
mbta-delay-prediction/
├── src/
│   ├── collect_transitmatters.py   # Download real MBTA data from TransitMatters API
│   ├── collect_weather.py          # Download Boston weather from Open-Meteo
│   ├── clean_data.py               # Merge, compute delays, clean data
│   ├── features.py                 # Feature engineering
│   ├── train.py                    # Model training + evaluation
│   ├── evaluate.py                 # Deep evaluation (ROC/PR curves, slicing)
│   └── visualize.py                # Generate plots
├── tests/
│   └── test_pipeline.py            # Unit + integration tests
├── data/
│   ├── raw/
│   │   ├── travel_times.csv        # 131,753 trip records (Jan–Oct 2025)
│   │   └── weather.csv             # Daily Boston weather (Jan–Oct 2025)
│   └── processed/
│       ├── clean.csv               # Cleaned and merged data
│       ├── features.csv            # Feature matrix
│       ├── model_results.csv       # Validation F1 per model
│       ├── best_model.pkl          # Saved model + scaler
│       ├── val_predictions.csv     # Best model validation predictions
│       ├── test_predictions.csv    # Best model test predictions
│       └── plots/                  # Generated visualizations
├── Project_Description.md
├── requirements.txt
└── README.md
```

---

## Progress Against Project Timeline

| Week | Goal | Status |
|------|------|--------|
| 1 | API setup, route/trip metadata extraction | ✅ Complete |
| 2 | Data collection, delay computation, weather merge | ✅ Complete (TransitMatters + Open-Meteo) |
| 3 | Pipeline hardening, EDA, outlier handling | ✅ Complete |
| 4 | Baseline Logistic Regression, March check-in | ✅ Complete |
| 5 | Feature engineering, Random Forest, GBT | ✅ Complete |
| 6 | Error analysis, feature importance plots, PR/ROC curves | ✅ Complete |
| 7 | Robustness checks, model selection, documentation | 🔄 In progress |
| 8 | Final polish, report, 10-min presentation video | ⬜ Pending |

---

## Environment

- Python 3.9+
- macOS, Linux, or Windows (WSL recommended on Windows)
- All dependencies in `requirements.txt`

---

## Limitations

- Data covers January–October 2025 only; Routes 1, 15, 39, 57 drop off in the TransitMatters archive after October 2025.
- Weather is merged at daily granularity — hourly weather would improve accuracy.
- Only one stop-pair per route is tracked (inbound main segment); delays at other stops or on outbound trips are not captured.
- The validation–test F1 gap suggests the tuned threshold does not generalize perfectly across seasons; a rolling or cross-validated threshold would help.
- Model performance is limited by the 12 available features; vehicle headway and real-time crowding data would likely improve prediction.
