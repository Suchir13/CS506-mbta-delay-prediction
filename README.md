# Predicting MBTA Bus Delays Using Weather and Time Features
**BU CS 506 Final Project**

> 🎥 **Presentation Video:** [Link to be added after recording]

---

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project predicts whether an MBTA bus will arrive **more than 5 minutes late** using historical bus schedule/prediction data combined with Boston weather conditions and time-based features.

Understanding delay patterns helps identify factors contributing to unreliable service and provides insights for transit optimization.

---

## How to Build and Run

### 1. Prerequisites
- Python 3.9 or newer
- `pip`

### 2. Clone and Install
```bash
git clone https://github.com/YOUR_USERNAME/mbta-delay-prediction.git
cd mbta-delay-prediction
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

### 5. (Optional) Re-collect fresh data
Only needed if you want to download new MBTA/weather data:
```bash
# Get free keys first:
# MBTA: https://api-v3.mbta.com/
# NOAA: https://www.ncdc.noaa.gov/cdo-web/token
cp .env.example .env  # then fill in your keys
python src/collect_mbta.py
python src/collect_weather.py
```

---

## Project Goals

**Primary Goal:** Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, day of week) most strongly influence delays.
- Visualize delay patterns across routes, times, and weather conditions.

---

## Data Collection

### MBTA v3 API
Bus schedule and prediction data are collected from the [MBTA v3 API](https://api-v3.mbta.com/).

| Endpoint | What it provides |
|----------|-----------------|
| `/routes` | Route IDs and names |
| `/schedules` | Planned arrival times per stop |
| `/predictions` | Real-time estimated arrival times |

**Target routes:** 1, 15, 28, 39, 57 (high-traffic Boston corridors)

**Delay formula:** `delay = predicted_arrival − scheduled_arrival` (in minutes)

### NOAA Weather Data
Historical daily weather from [NOAA CDO API](https://www.ncdc.noaa.gov/cdo-web/).
- **Station:** Boston Logan Airport (USW00014739)
- **Variables:** TMAX, TMIN, PRCP (precipitation), SNOW, SNWD, AWND (wind speed)
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
| Weather imputation | Fill missing weather values with monthly median |
| Temporal consistency | Merge weather by date only (no future data leakage) |

---

## Feature Extraction

Features used in the model:

| Feature | Description |
|---------|-------------|
| `hour` | Hour of scheduled arrival (0–23) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_peak` | 1 if weekday 7–9 AM or 4–7 PM |
| `route_encoded` | Numeric encoding of route ID |
| `route_avg_delay` | Historical average delay for the route |
| `is_rainy` | 1 if precipitation > 0.1 inches |
| `is_snowy` | 1 if snowfall > 0.1 inches |
| `TMAX` / `TMIN` | Daily high/low temperature (°F) |
| `PRCP` | Daily precipitation (inches) |
| `SNOW` | Daily snowfall (inches) |
| `AWND` | Average wind speed (mph) |

---

## Modeling

Three classifiers are trained and compared in `src/train.py`:

1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — captures non-linear interactions
3. **Gradient Boosted Trees** — typically best performance

**Split strategy (time-based, no shuffle):**
- Train: first 70% of data
- Validation: next 15%
- Test: final 15%

**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

The best model (by validation F1) is saved to `data/processed/best_model.pkl`.

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Logistic Regression (baseline) | 0.322 | 0.487 | 0.630 |
| **Random Forest (best)** | **0.675** | **0.544** | **0.657** |
| Gradient Boosted Trees | 0.670 | 0.525 | 0.642 |

**Key Findings:**
- Top delay predictors: is_peak (51%) and hour (43%) — time of day dominates
- Peak hours (7–9 AM, 4–7 PM weekdays) show significantly higher delay rates
- Weather features had low importance with current data (delays are simulated)
- Random Forest outperformed Logistic Regression and GBT on F1 and ROC-AUC

---

## Visualizations

All plots are saved to `data/processed/plots/` after running `make visualize`.

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
- End-to-end delay computation from raw schedule + prediction records

```bash
make test
# or
pytest tests/ -v
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs these automatically on every push.

---

## Repository Structure

```
mbta-delay-prediction/
├── src/
│   ├── collect_mbta.py      # Download MBTA schedule + prediction data
│   ├── collect_weather.py   # Download NOAA weather data
│   ├── clean_data.py        # Merge, compute delays, clean data
│   ├── features.py          # Feature engineering
│   ├── train.py             # Model training + evaluation
│   └── visualize.py         # Generate plots
├── tests/
│   └── test_pipeline.py     # Unit + integration tests
├── data/
│   ├── raw/                 # Raw downloaded data (git-ignored)
│   └── processed/           # Cleaned data, features, model (git-ignored)
├── .github/workflows/
│   └── tests.yml            # CI: run tests on push
├── .env.example             # API key template
├── requirements.txt
├── Makefile
└── README.md
```

---

## Environment

- Python 3.9+
- macOS, Linux, or Windows (WSL recommended on Windows)
- All dependencies in `requirements.txt`

---

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for any new functionality
4. Run `make test` to make sure everything passes
5. Open a pull request

---

## Limitations

- MBTA predictions API has limited historical depth; data collection should run continuously over weeks for a large dataset.
- Weather is merged at daily granularity — hourly weather would improve accuracy.
- Route-level average delay is computed globally (not per time period), which could leak mild signal; documented as a known limitation.
- Model performance depends heavily on dataset size; results with < 1,000 rows may not generalize.