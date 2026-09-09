# Adaptive Hospital Demand Forecasting

## Overview

Hospital demand can change over time due to seasonal patterns, changing patient behaviour, holidays, outbreaks, weather conditions, and unexpected events. A forecasting model trained only on historical patterns may gradually become less reliable when these patterns change.

This project develops an **AI/ML-based adaptive time-series forecasting system** for hospital demand that not only predicts future demand but also continuously monitors forecasting performance, detects meaningful changes in behaviour, and adapts the forecasting strategy when required.

---

## Problem Statement

**Adaptive Time-Series Forecasting Under Changing Conditions**

The system must generate forecasts using only information available before the corresponding actual observation is revealed. It must continuously monitor whether the learned forecasting patterns remain reliable and adapt when a meaningful change is detected, while avoiding unnecessary adaptation to normal fluctuations or short-lived anomalies.

---

## Proposed Solution

Our solution follows an adaptive forecasting pipeline:

```text
Historical Observations
          ↓
   Feature Processing
          ↓
   Forecasting Model
          ↓
    Future Prediction
          ↓
 Actual Observation Arrives
          ↓
    Forecast Error
          ↓
 Monitoring / Change Score
          ↓
   Change Detection
       ↙       ↘
     No         Yes
     ↓           ↓
 Continue     Adapt Model
 Monitoring       ↓
              Updated Model
                   ↓
             New Forecast
```

The system therefore combines **forecasting, continuous monitoring, change detection, and model adaptation** within a single pipeline.

---

## Application Domain

### Hospital Demand Forecasting

The system focuses on forecasting hospital demand from time-series observations.

Potential applications include:

* Patient demand forecasting
* Appointment volume forecasting
* Emergency visit forecasting
* Resource planning
* Capacity management

The forecasting system is designed to respond when the underlying demand behaviour changes significantly.

---

## Current Approach

The initial implementation uses a **Random Forest regression model** as the forecasting baseline.

The system maintains chronological ordering of observations and avoids shuffling the time-series data.

The initial pipeline includes:

1. Data loading and validation
2. Time-series preprocessing
3. Forecast generation
4. Actual observation comparison
5. Forecast error calculation
6. Change-score monitoring
7. Adaptive model retraining
8. Dashboard visualization

The forecasting and adaptation strategy will be further refined and evaluated using the organizer's evaluation data and real-time evaluation interface.

---

## 🏗️ System Architecture

The overall architecture consists of:

* Data processing layer
* Forecasting layer
* Monitoring layer
* Change detection layer
* Adaptation layer
* Backend/API layer
* Dashboard and visualization layer

### Architecture Diagram

> **Add the project architecture image here.**

Save your architecture image inside:

```text
docs/architecture.png
```

Then use:

```markdown
![System Architecture](docs/architecture.png)
```

---

## 📊 Dashboard

The dashboard is designed to provide an end-to-end view of the forecasting system.

It displays:

* **Current Forecast**
* **Actual Observation**
* **Forecast Error**
* **Change / Uncertainty Score**
* **System Status**
* **Adaptation Status**
* Forecast vs Actual visualization
* Recent observations and predictions

The dashboard will also be integrated with the organizer's evaluation API for real-time evaluation.

---

## 🔄 Adaptive Forecasting Workflow

The system continuously follows this process:

```text
1. Receive available historical observations
            ↓
2. Generate the next forecast
            ↓
3. Wait for the actual observation
            ↓
4. Calculate forecast error
            ↓
5. Monitor forecasting behaviour
            ↓
6. Calculate change score
            ↓
7. Determine whether meaningful change exists
            ↓
8. If required, adapt/retrain the model
            ↓
9. Generate the next forecast
```

No future observation is used to produce an earlier prediction.

---

## 🔐 Data Leakage Prevention

Because this is a time-series forecasting problem, the system preserves chronological ordering.

For each prediction, only information available before the corresponding actual observation is used.

The system does not use:

* Future target values
* Future observations
* Hard-coded change timestamps
* Hard-coded row indices as adaptation triggers

---

## 🛠️ Technology Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Plotly**

---

## 📁 Project Structure

```text
adaptive-hospital-forecasting/
│
├── data/
│   └── training_set.csv
│
├── src/
│   ├── data_loader.py
│   ├── forecaster.py
│   ├── monitor.py
│   ├── adapter.py
│   └── pipeline.py
│
├── dashboard/
│   └── app.py
│
├── logs/
├── tests/
│
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Current Development Status

### Completed

* [x] Project structure
* [x] Training dataset integration
* [x] Time-series data loading
* [x] Initial forecasting pipeline
* [x] Forecast error calculation
* [x] Monitoring mechanism
* [x] Initial adaptation structure
* [x] Dashboard prototype
* [x] System architecture

### In Progress

* [ ] Robust change detection
* [ ] Improved adaptive strategy
* [ ] Real-time evaluation API integration
* [ ] Evaluation logging
* [ ] End-to-end testing
* [ ] Deployment and final optimization

---

## 🔮 Future Development

The system can be further enhanced through:

* More robust behavioural change detection
* Improved adaptation strategies
* Uncertainty estimation
* Multiple forecasting models
* Automatic model selection
* Real-time monitoring
* Improved hospital resource forecasting

---

## 👥 Team

**Team:** Neural Ninjas

**Hackathon:** X'O Code 2026

**Domain:** AI/ML

**Problem Statement:** PS02 — Adaptive Time-Series Forecasting Under Changing Conditions
