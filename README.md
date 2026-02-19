# Predicting MBTA Bus Delays Using Weather and Time Features
BU CS506 Final Project

## Project Description

Public transportation reliability is critical for daily commuters in Boston. This project aims to predict whether an MBTA bus will experience a significant delay (e.g., more than 5 minutes late) using historical bus arrival data, weather conditions, and time-based features.

Understanding delay patterns can help identify factors contributing to unreliable service and provide insights for transit optimization.

---

## Project Goals

**Primary Goal:**
- Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

**Secondary Goals:**
- Identify which factors (e.g., rainfall, temperature, time of day, day of week) most strongly influence delays.
- Visualize patterns in delays across routes, times, and weather conditions.

---

## Data Collection Plan

### 1. MBTA v3 API (Historical Bus Arrival and Schedule Data)
Bus arrival, route, and trip data will be collected using the [MBTA v3 API](https://api-v3.mbta.com/).
- **Routes:** Route information (bus lines, route IDs, and names) will be retrieved from the [`/routes` endpoint](https://api-v3.mbta.com/docs/swagger/index.html#/Route/ApiWeb_RouteController_index).
- **Trips:** Trip-level data representing individual bus runs will be obtained from the `/trips` endpoint.
- **Schedules:** Scheduled arrival times for buses will be collected from the `/schedules` endpoint.
- **Predictions:** Actual arrival times will be retrieved from the `/predictions` endpoint.

Bus delays will be computed by comparing actual arrival times with scheduled arrival times, where delay is defined as:  
`delay = predicted arrival time − scheduled arrival time`

### 2. Weather Data (Historical Weather Conditions for Boston)
Historical weather data will be obtained from [NOAA Climate Data Online (CDO)](https://www.ncdc.noaa.gov/cdo-web/).
- The Daily Summaries (GHCN-Daily) dataset will be used to collect daily weather variables including maximum and minimum temperature, precipitation, and snowfall.
- Weather observations will be retrieved in CSV format from the Boston Logan International Airport weather station (USW00014739) for the same date range as the MBTA bus data.
- Weather data will be merged with bus arrival data based on the date to analyze the impact of weather conditions on bus delays.

*Note: The [OpenWeather API](https://openweathermap.org/api) is listed as an alternative data source if higher temporal resolution or supplementary weather attributes are required.*

### 3. Optional: Public Holiday Calendar Data
Public holiday information for Massachusetts will be obtained from [Office Holidays](https://www.officeholidays.com/countries/usa/massachusetts).
- Holiday dates will be used to create a binary holiday indicator feature.
- This feature will help analyze whether public holidays are associated with increased or decreased bus delays compared to regular weekdays.

---

## Data Cleaning Plan

We will document and implement these cleaning steps in code (not manual edits):

### MBTA Data Cleaning
- **Filter missing keys:** Drop rows with missing required keys (route_id, stop_id, scheduled_time).
- **Format datetime:** Resolve timestamp formats and time zones into a consistent `datetime`.
- **Deduplication:** Remove duplicate events (same trip/stop timestamp) using a deterministic rule.
- **Validate delays:**
  - Keep early arrivals as negative delay (or optionally clamp at 0 for the “late” definition—decision will be documented).
  - Flag extreme outliers (e.g., > 2 hours late) and decide whether to cap or exclude.

### Weather Data Cleaning
- **Handle missing NOAA fields with:**
  - Simple imputation (median for that month) **or**
  - “Missing flag” features to preserve missingness signal.

### Merge Logic (Avoid Leakage)
- Merge on service date for daily NOAA weather.
- If hourly weather is used later, merge on the nearest prior timestamp (documented).

---

## Feature Extraction Plan

Planned features include:
- Hour of day
- Day of week
- Weekend vs weekday
- Temperature
- Rainfall
- Snow indicators
- Route ID
- Historical average delay for route
- Binary holiday indicator (Optional)

---

## Modeling Plan (Preliminary)

We plan to begin with:
- Logistic Regression (baseline)

Then experiment with:
- Random Forest
- Gradient Boosted Trees (GBDT)

---

## Evaluation & Test Plan

### Data Splitting Strategy (Avoid Temporal Leakage)
**Primary plan:** Train on earlier dates, test on later dates (time-based split). 
- **Train:** First ~70–80% of timeline
- **Validation:** Next ~10–15%
- **Test:** Final ~10–15%

We will also sanity-check generalization by evaluating performance across:
- Different routes
- Peak vs. off-peak time windows
- Weather vs. non-weather days

### Metrics
- **Accuracy** (for quick reference)
- **Precision, Recall, F1** (delay is usually the minority class)
- **ROC-AUC and/or PR-AUC**
- **Confusion matrix** at a chosen threshold

---

## Visualization Plan

Planned visualizations include:
- Delay frequency by hour of day
- Delay frequency by route
- Weather vs delay scatter plots
- Feature importance plots
- Confusion matrix and ROC/PR curves

---

## Timeline

**Week 1: API Setup and Scoping**
- Select specific high-traffic MBTA bus routes and define the historical date range to ensure a manageable but representative dataset.
- Develop a Python wrapper for the MBTA v3 API, including local caching mechanisms to avoid hitting rate limits during iterative testing.
- Extract route and trip metadata from the `/routes` and `/trips` endpoints and validate the data schema.

**Week 2: Data Collection and Merging**
- Write and execute automated scripts to fetch scheduled and actual arrival times from the `/schedules` and `/predictions` endpoints.
- Calculate the primary target variable (delay in minutes) and save the initial raw and processed datasets.
- Download NOAA GHCN-Daily weather data for Boston Logan Airport and build a prototype script to merge weather features with bus data based on service dates without introducing temporal leakage.

**Week 3: Pipeline Hardening and EDA**
- Formalize the data cleaning pipeline in code: drop invalid rows, standardize datetime formats, handle missing weather values, and apply deterministic rules to remove duplicates.
- Address extreme outliers (e.g., buses >2 hours late) based on documented rules.
- Conduct initial Exploratory Data Analysis (EDA) by generating plots to observe delay distributions across different hours of the day, days of the week, and routes.

**Week 4 (March Check-in Target): Baselines and Preliminary Review**
- Train a baseline Logistic Regression model to establish a performance floor.
- Evaluate the baseline model on the validation set using our defined classification metrics (Accuracy, Precision, Recall, F1).
- Prepare for the March check-in by compiling preliminary EDA visualizations and documenting the end-to-end data collection and cleaning pipeline.

**Week 5: Feature Engineering and Advanced Modeling**
- Engineer more complex, predictive features: rolling average delay statistics, proxy measures for bus headways, and optimized encodings for categorical variables.
- Begin experimenting with non-linear, tree-based models, specifically Random Forest and Gradient Boosted Trees (GBDT).
- Outline and execute a hyperparameter tuning strategy for these advanced models.

**Week 6: Deep Evaluation and Visual Interpretation**
- Deepen model evaluation by conducting error analysis (e.g., analyzing false positives vs. false negatives).
- Enhance project visualizations by generating feature importance charts to interpret which factors drive model decisions.
- Plot Precision-Recall (PR) curves and confusion matrices optimized for our specific >5 minutes late threshold.

**Week 7 (April Check-in Target): Robustness and Selection**
- Conduct robustness checks to ensure the model generalizes well across unseen routes, peak vs. off-peak hours, and extreme vs. mild weather days.
- Select the final best-performing model for the April check-in.
- Thoroughly document any identified limitations, edge cases, and areas where the model struggles to predict delays accurately.

**Week 8: Final Polish and Presentation**
- Finalize the project repository and `README.md` to ensure full reproducibility (ensuring clean data-loading scripts, requirements, and instructions).
- Polish all final figures, evaluation results, and interpretations for the final written report.
- Prepare, practice, and record the required 10-minute final presentation, adding the video link to the repository README.
