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
# Place MassGIS monthly CSVs in data/raw/arrival_departure/ then:
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

### 5. Try the Interactive Predictor
Enter a route, hour, day, and weather condition to get a live delay prediction from the trained model:
```bash
python src/predict.py
```

### 6. (Optional) Deep Evaluation
Generates PR curves, ROC curves, and per-route/per-hour performance breakdowns:
```bash
python src/evaluate.py
```

### 7. (Optional) Re-collect fresh data
No API keys required for either source:
```bash
# To re-collect data, download monthly CSVs from:
# 2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
# 2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c
# Place in data/raw/arrival_departure/ and run python src/clean_data.py

# Re-collect Boston weather from Open-Meteo (free, no auth)
python src/collect_weather.py --start 2025-01-01 --end 2025-10-31
```

### 8. (Optional) Use Official MassGIS Dataset
For higher-fidelity data with per-stop actual arrival times and direction info:
```bash
# Download monthly CSVs from:
# 2025: https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
# 2026: https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c
# Place files in data/raw/arrival_departure/ then run:
python src/clean_data.py --source official --dataset-dir data/raw/arrival_departure
python src/features.py
python src/train.py
python src/visualize.py
```

---

## Project Goals

**Primary Goal:** Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (rainfall, temperature, time of day, route) most strongly influence delays.
- Visualize delay patterns across routes, times, and weather conditions.
- Provide an interactive predictor so users can query the model directly.

---

## Data Collection

### Source 1 — Boston Weather: Open-Meteo
Historical daily weather from [Open-Meteo](https://archive-api.open-meteo.com/v1/archive). No API key required.

- **Location:** Boston (42.3601 N, 71.0589 W)
- **Variables:** TMAX, TMIN, PRCP (precipitation), SNOW, AWND (wind speed)
- **Units:** °F, inches, mph
- Merged with MBTA data by service date
- **Dataset:** 304 days, January 2025 – October 2025

### Source 2 — Official MassGIS / MBTA Dataset (also supported)
The official MBTA Bus Arrival/Departure dataset published by MassDOT provides per-stop actual vs scheduled arrival times for all routes back to 2019. This is the primary upstream source that TransitMatters derives data from.

- **2025 dataset:** https://gis.data.mass.gov/datasets/924df13d845f4907bb6a6c3ed380d57a/about
- **2026 dataset:** https://gis.data.mass.gov/datasets/9d8a8cad277545c984c1b25ed10b7d3c
- 35+ million records with `scheduled` and `actual` timestamps per stop
- Includes direction (Inbound/Outbound), headway, and timepoint type
- `clean_data.py` supports this via `--source official`

---

## Data Cleaning

All cleaning logic is in `src/clean_data.py` — no manual edits anywhere.

| Step | Action |
|------|--------|
| Drop missing keys | Remove rows missing `route_id`, `date`, or arrival time |
| Normalize fields | Standardize date format, strip whitespace |
| Deduplication | Keep first occurrence per (trip, stop, date) |
| Delay computation | `delay_minutes = actual − benchmark` (TransitMatters) |
| Outlier flagging | Flag \|delay\| > 120 min — kept but marked `is_outlier=1` |
| Weather imputation | Fill missing weather values with column median |
| Weather merge | Join on service date — no future data leakage |

---

## Feature Extraction

Features built in `src/features.py` (15 total):

| Feature | Description |
|---------|-------------|
| `hour` | Hour of scheduled arrival (0–23) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_peak` | 1 if weekday 7–9 AM or 4–7 PM |
| `route_encoded` | Numeric encoding of route ID |
| `has_actual` | 1 if actual arrival time was recorded |
| `scheduled_headway_minutes` | Planned gap between buses (minutes) |
| `scheduled_headway_missing` | 1 if headway data was absent |
| `is_rainy` | 1 if precipitation > 0.1 inches |
| `is_snowy` | 1 if snowfall > 0.1 inches |
| `TMAX` / `TMIN` | Daily high/low temperature (°F) |
| `PRCP` | Daily precipitation (inches) |
| `SNOW` | Daily snowfall (inches) |
| `AWND` | Average wind speed (mph) |

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

**Hyperparameter tuning:** `src/randomForestParams.py` sweeps multiple Random Forest configurations (n_estimators, max_depth, min_samples). The compact configuration (n=30, depth=16) gives the best storage/performance tradeoff at ~38MB vs ~1GB for the largest variant with minimal F1 gain.

---

## Results

### Validation Set

| Model | Val F1 | Threshold |
|-------|--------|-----------|
| Logistic Regression (baseline) | 0.405 | 0.42 |
| **Random Forest (best)** | **0.422** | **0.53** |
| Gradient Boosted Trees | 0.427 | 0.64 |

### Test Set (best model: Random Forest)

| Metric | Score |
|--------|-------|
| Accuracy | 0.352 |
| Precision | 0.152 |
| Recall | 0.765 |
| F1 | 0.254 |
| ROC-AUC | 0.544 |
| PR-AUC | 0.157 |

### Per-Slice Performance (Test Set, from `evaluate.py`)

| Slice | Accuracy | F1 |
|-------|----------|----|
| Peak hours | 0.235 | 0.314 |
| Off-peak hours | 0.407 | 0.212 |
| Route 1 (encoded=3) | 0.434 | 0.438 |
| Route 28 (encoded=4) | 0.351 | 0.251 |

**Key Findings:**
- `route_encoded` is the strongest predictor (58.7% importance) — with real data, different routes have genuinely different delay rates
- Weather features (TMAX, TMIN, AWND, PRCP) contribute meaningfully — ~22% combined importance
- High Recall (0.765) means the model catches ~77% of real delays — useful for commuters
- Validation–test F1 gap reflects seasonal threshold shift — tuned on summer/fall, tested on later fall with different delay distribution

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

`src/predict.py` loads the trained model and lets you query it interactively:

```
$ python src/predict.py

=== MBTA Delay Predictor ===

Type 'quit' at any prompt to exit.

Enter hour of day (0 to 23): 8
Enter day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun): 2

Available routes: 01, 04, 07, 08, 09, 10, 100, 101, 104, 105, 106, 108, 109, 11, 110, 111, 112, 114, 116, 119, 120, 121, 131, 132, 134, 137, 14, 15, 16, 17, 171, 18, 19, 191, 192, 193, 194, 201, 202, 21, 210, 211, 215, 216, 217, 22, 220, 222, 225, 226, 23, 230, 236, 238, 24, 240, 245, 26, 28, 29, 30, 31, 32, 33, 34, 34E, 35, 350, 351, 354, 36, 37, 38, 39, 40, 41, 411, 42, 424, 426, 428, 429, 43, 430, 435, 436, 439, 44, 441, 442, 45, 450, 451, 455, 456, 47, 50, 501, 504, 505, 51, 52, 55, 553, 554, 556, 558, 57, 59, 60, 600, 61, 62, 64, 65, 66, 67, 68, 69, 70, 7001, 71, 712, 713, 73, 74, 743, 746_, 75, 76, 77, 78, 80, 8007, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, CT2, CT3, SL1, SL2, SL3, SL4, SL5, rad
Enter route no: 230
Direction ['Inbound', 'Outbound'] [default=Outbound]: Inbound
Point type ['Endpoint', 'Midpoint', 'Startpoint'] [default=Midpoint]: Midpoint
Weather condition (clear/ rain/ snow) [default=clear]: snow

=== RESULT ===
Delay probability: 0.137
Prediction: Likely ON TIME!!

Why this prediction?
- Peak hours increase delays
- Snow increases delays
```

> Note: The model is biased towards predicting on-time because the dataset is imbalanced (82% on-time vs 18% delayed). High Recall means real delays are still caught ~77% of the time.

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
│   ├── collect_transitmatters.py   # Download real MBTA data from TransitMatters
│   ├── collect_weather.py          # Download Boston weather from Open-Meteo
│   ├── clean_data.py               # Merge, compute delays, clean data
│   ├── features.py                 # Feature engineering (15 features)
│   ├── train.py                    # Model training + evaluation
│   ├── evaluate.py                 # PR/ROC curves, per-route/peak slicing
│   ├── randomForestParams.py       # Random Forest hyperparameter sweep
│   ├── predict.py                  # Interactive delay predictor
│   └── visualize.py                # Generate 5 EDA/results plots
├── tests/
│   └── test_pipeline.py            # 12 unit + integration tests
├── data/
│   ├── raw/
│   │   ├── travel_times.csv        # 131,753 real trip records (Jan–Oct 2025)
│   │   └── weather.csv             # 304 days Boston weather (Jan–Oct 2025)
│   └── processed/
│       ├── clean.csv               # Cleaned and merged dataset
│       ├── features.csv            # Final feature matrix (15 features)
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
| 2 | Real data collection, weather merge | ✅ Complete — TransitMatters + Open-Meteo |
| 3 | Pipeline hardening, EDA, outlier handling | ✅ Complete |
| 4 | Baseline Logistic Regression, March check-in | ✅ Complete |
| 5 | Feature engineering, Random Forest, GBT, hyperparameter tuning | ✅ Complete |
| 6 | Error analysis, feature importance, PR/ROC curves | ✅ Complete |
| 7 | Robustness checks (peak/off-peak, per-route), interactive predictor | ✅ Complete |
| 8 | Final polish, 10-min presentation video | ⬜ Pending |

---

## Environment

- Python 3.9+
- macOS, Linux, or Windows (WSL recommended on Windows)
- All dependencies in `requirements.txt`

---

## Limitations

- Only one stop-pair per route tracked in TransitMatters data — inbound main segment only
- Weather merged at daily granularity — hourly weather would improve signal
- Validation–test F1 gap (0.422 → 0.254) suggests threshold doesn't generalise across seasons
- Dataset imbalance (82% on-time) means precision is low — model favours recall
- MassGIS official dataset (35M+ rows, all stops, all routes) would significantly improve coverage — pipeline already supports it via `--source official`
