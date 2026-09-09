# Adaptive Hospital Demand Forecasting

An adaptive time-series forecasting system designed to forecast hospital demand and monitor forecasting performance under changing data behaviour.

## Project Overview

The system uses historical time-series data to train a forecasting model and generate sequential predictions.

The project is designed to evolve through multiple stages:

- Version 1: Baseline forecasting and monitoring
- Version 2: Gradual behavioural change detection
- Version 3: Temporary anomaly and false-alarm handling

## Version 1

The current version provides:

- Data preprocessing
- Forecasting model
- Time-based model validation
- Sequential prediction simulation
- Forecast error monitoring
- Streamlit dashboard

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Structure

```text
adaptive-hospital-forecasting/
├── data/
├── models/
├── logs/
├── src/
├── dashboard/
├── train.py
├── run_local.py
├── requirements.txt
└── README.md