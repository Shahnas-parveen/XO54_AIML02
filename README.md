# 🏥 Adaptive Hospital Demand Forecasting

An adaptive time-series forecasting system that predicts hospital demand, detects behavioural changes, handles temporary anomalies, and adapts the model only when sufficient evidence is available.

## 🎯 Problem Statement

Hospital demand can change over time because of changing patient patterns, seasonal effects, operational conditions, and unexpected events.

A model trained only on historical data may become inaccurate when demand behaviour changes. However, a sudden temporary disturbance should not cause unnecessary model retraining.

This project addresses both situations using forecasting, error monitoring, change detection, anomaly detection, and controlled model adaptation.

## 💡 Proposed Solution

The system follows this pipeline:

**Historical Data → Feature Engineering → Forecast → Actual Target → Error Monitoring → Change Detection → Hold / Adapt Model**

The system handles two evaluation scenarios.

### 🔵 Challenge 1 — The Quiet Shift

The system detects gradual and sustained changes in forecasting behaviour.

It compares recent forecasting errors with a historical baseline using:

- Historical error baseline
- Recent rolling error
- Error ratio
- Error trend
- Multiple confirmations

Decision flow:

**NORMAL → WATCH → CONFIRMED CHANGE → MODEL ADAPTED**

The model is adapted only after sustained evidence of a behavioural change.

### 🟣 Challenge 2 — The False Alarm

The system detects a sudden unusual deviation while avoiding unnecessary model adaptation.

Decision flow:

**NORMAL → TEMPORARY ANOMALY → RECOVERY WATCH → RECOVERED → NORMAL**

During a temporary disturbance, the system keeps the existing model:

**MODEL ACTION → HOLD MODEL**

If a deviation persists without recovery, the system can instead confirm a change and adapt the model.

## 🧠 Forecasting Model

The forecasting model uses **Ridge Regression with feature scaling**.

The model uses six original signals:

- `feature_1`
- `feature_2`
- `feature_3`
- `feature_4`
- `feature_5`
- `feature_6`

Additional features are created from the timestamp and previous target values:

- Day of week
- Month
- Day of year
- Lag 1, 2, 3, 7 and 14
- Rolling mean 3, 7 and 14

These features help the model capture recent demand behaviour and time-based patterns.

## 📊 Dataset

The training dataset contains:

- **550 observations**
- **6 input features**
- **1 target variable**
- Daily timestamp-based observations

The data is sorted chronologically and an 80/20 time-ordered split is used for validation.

## 📈 Model Results

The improved forecasting model achieved:

| Metric | Result |
|---|---:|
| MAE | 2.1987 |
| RMSE | 2.7886 |
| MAPE | 18.00% |
| R² | 0.7471 |

The final trained model is saved as:

`models/forecasting_model.pkl`

## 🔍 Detection and Adaptation

### Quiet Shift Detection

The detector monitors whether recent forecasting errors remain consistently above the historical baseline and whether the error trend indicates a sustained change.

This prevents a single unusual prediction from triggering unnecessary retraining.

### False Alarm Detection

The anomaly detector identifies sudden deviations and temporarily holds the model while observing subsequent observations.

If forecasting error returns toward the normal baseline, the event is treated as temporary and the existing model is retained.

This creates a conservative adaptation strategy:

**Detect → Observe → Confirm → Adapt only when necessary**

## 🔄 Sequential Evaluation

The forecasting pipeline processes observations sequentially:

**Get Next Observation → Generate Forecast → Submit Prediction → Receive Actual Target → Calculate Error → Update Target History → Run Detector → Hold or Adapt Model**

Actual target values revealed after each prediction are added to the target history so that future lag and rolling features can be updated.

## 🖥️ Dashboard

The project includes an interactive **Streamlit dashboard** showing:

- Forecast vs Actual
- Forecast error
- Average error
- Change detection status
- Baseline and recent error
- Error trend
- Anomaly and recovery behaviour
- Model action
- Recent evaluation records

The dashboard provides separate views for:

- **Challenge 1 — The Quiet Shift**
- **Challenge 2 — The False Alarm**

Run the dashboard locally using:

`streamlit run dashboard/app.py`

## 🌐 Evaluation API

The system is designed for sequential evaluation using the provided evaluation API.

The required interaction is:

**GET next → Generate prediction → POST prediction → Receive actual → Continue**

Separate sessions are used for the two challenge streams.

Session identifiers are kept local and are not included in the repository.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Ridge Regression
- Joblib
- Requests
- Streamlit
- Plotly
- Git & GitHub

## 📁 Project Structure

    XO54_AIML02/
    │
    ├── dashboard/
    │   └── app.py
    ├── data/
    │   └── training_set.csv
    ├── logs/
    │   ├── sc1_log.csv
    │   └── sc2_log.csv
    ├── models/
    │   └── forecasting_model.pkl
    ├── demo_data/
    │   ├── sc1_demo.csv
    │   └── sc2_demo.csv
    ├── src/
    │   ├── __init__.py
    │   ├── adaptation.py
    │   ├── anomaly_detector.py
    │   ├── api_client.py
    │   ├── change_detector.py
    │   ├── features.py
    │   ├── model.py
    │   ├── monitor.py
    │   ├── pipeline.py
    │   └── preprocessing.py
    ├── run_demo.py
    ├── run_evaluation.py
    ├── train.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## 🚀 Running the Project

### Install dependencies

`pip install -r requirements.txt`

### Train the model

`python train.py`

### Run the local demonstrations

`python run_demo.py`

This generates demonstration results for both challenge scenarios and creates the corresponding evaluation logs.

### Launch the dashboard

`streamlit run dashboard/app.py`

## 🎯 Key Highlights

- Sequential time-series forecasting
- Lag and rolling feature engineering
- Behavioural change detection
- Temporary anomaly detection
- Evidence-based model adaptation
- Conservative false-alarm handling
- Interactive monitoring dashboard
- API-ready sequential evaluation pipeline

## 👥 Team

**Team Name:** Neural Ninjas

**Project:** Adaptive Hospital Demand Forecasting

**Challenge:** PS02 — Adaptive Time-Series Forecasting

## 📚 References

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly

## ⭐ Conclusion

Adaptive Hospital Demand Forecasting combines machine learning forecasting with behavioural monitoring and controlled model adaptation.

Instead of treating every increase in forecasting error as a permanent change, the system distinguishes between normal variation, gradual behavioural change, and temporary anomalies.

The result is a forecasting system that is **adaptive when necessary, stable when appropriate, and transparent in its decisions**.