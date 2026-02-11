# Predicting MBTA Bus Delays Using Weather and Time Features
BU CS506 Final Project

## Project Description

Public transportation reliability is critical for daily commuters in Boston. 
This project aims to predict whether an MBTA bus will experience a significant delay 
(e.g., more than 5 minutes late) using historical bus arrival data, weather conditions, 
and time-based features.

Understanding delay patterns can help identify factors contributing to unreliable service 
and provide insights for transit optimization.

---

## Project Goals

Primary Goal:
- Predict whether a bus arrival will be delayed by more than 5 minutes using weather and time-based features.

Secondary Goals:
- Identify which factors (e.g., rainfall, temperature, time of day, day of week) most strongly influence delays.
- Visualize patterns in delays across routes, times, and weather conditions.

Evaluation:
- Use classification metrics such as accuracy, precision, recall, and ROC-AUC.
- Compare baseline model (logistic regression) with more advanced models (e.g., random forest).

---

## Data Collection Plan

### Data Sources

1. MBTA API (Historical bus arrival and schedule data)
2. Weather data (NOAA or OpenWeather API)
3. Optional: Public holiday calendar data

### Collection Method

- Use Python scripts to query MBTA API for bus arrival and delay information.
- Use a weather API to retrieve historical weather data for the same timestamps.
- Merge datasets based on date and time.

All data collection will be automated through reproducible scripts.

---

## Data Cleaning Plan

- Remove incomplete or corrupted entries.
- Handle missing weather values.
- Convert timestamps to structured datetime features.
- Create delay threshold label (e.g., delayed > 5 minutes).

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

---

## Modeling Plan (Preliminary)

We plan to begin with:
- Logistic Regression (baseline)

Then experiment with:
- Random Forest
- Gradient Boosted Trees

We will compare models using cross-validation and hold-out test sets.

---

## Visualization Plan

Planned visualizations include:
- Delay frequency by hour of day
- Delay frequency by route
- Weather vs delay scatter plots
- Feature importance plots
- Confusion matrix and ROC curve

---

## Timeline

Weeks 1–2:
- Data collection and initial cleaning

Weeks 3–4:
- Feature engineering and exploratory data analysis

Weeks 5–6:
- Model development and evaluation

Weeks 7–8:
- Refinement, reproducibility setup (Makefile, tests), final report and presentation
