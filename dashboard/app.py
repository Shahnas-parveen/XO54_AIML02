import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from src.forecaster import PatientForecaster


DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "patient_data.csv"
)


st.set_page_config(
    page_title="Adaptive Hospital Demand Forecasting",
    layout="wide"
)

st.title("🏥 Adaptive Hospital Demand Forecasting")


# ==================================================
# LOAD WORKING DATA
# ==================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "patient_data.csv not found inside the data folder."
    )

    st.stop()


df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)


# ==================================================
# TRAIN MODEL
# ==================================================

model = PatientForecaster()

model.train(df)


# ==================================================
# NEXT DAY PREDICTION
# ==================================================

prediction = model.predict_next(df)

last_date = df["timestamp"].iloc[-1]

next_date = (
    last_date + pd.Timedelta(days=1)
)


st.header("Next-Day Patient Demand")


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted Patients",
        f"{prediction:.0f}"
    )

with col2:

    st.metric(
        "Prediction Date",
        next_date.strftime("%d %b %Y")
    )


st.info(
    f"Expected patient demand for "
    f"{next_date.strftime('%d %b %Y')}: "
    f"**{prediction:.0f} patients**"
)


# ==================================================
# ACTUAL PATIENT COUNT
# ==================================================

st.header("Enter Actual Patient Count")


actual_value = st.number_input(
    f"Actual patient count for "
    f"{next_date.strftime('%d %b %Y')}",
    min_value=0.0,
    step=1.0
)


# ==================================================
# EVALUATE + SAVE + RETRAIN
# ==================================================

if st.button("Submit Actual & Adapt Model"):

    # ----------------------------------------------
    # Calculate error
    # ----------------------------------------------

    error = abs(
        actual_value - prediction
    )

    percentage_error = (
        error / actual_value * 100
        if actual_value != 0
        else 0
    )


    # ----------------------------------------------
    # Create new observation
    # ----------------------------------------------

    new_row = pd.DataFrame({
        "timestamp": [next_date],
        "target": [actual_value]
    })


    # ----------------------------------------------
    # Add new observation to existing data
    # ----------------------------------------------

    updated_df = pd.concat(
        [
            df,
            new_row
        ],
        ignore_index=True
    )


    updated_df = (
        updated_df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )


    # ----------------------------------------------
    # SAVE TO CSV
    # ----------------------------------------------

    updated_df.to_csv(
        DATA_PATH,
        index=False
    )


    # ----------------------------------------------
    # RETRAIN MODEL
    # ----------------------------------------------

    model.train(updated_df)


    # ----------------------------------------------
    # NEW FORECAST
    # ----------------------------------------------

    new_prediction = (
        model.predict_next(updated_df)
    )

    new_last_date = (
        updated_df["timestamp"].iloc[-1]
    )

    new_next_date = (
        new_last_date
        + pd.Timedelta(days=1)
    )


    # ----------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------

    st.success(
        "Actual observation saved and model retrained."
    )


    st.subheader("Forecast Evaluation")


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted",
            f"{prediction:.0f}"
        )

    with col2:

        st.metric(
            "Actual",
            f"{actual_value:.0f}"
        )

    with col3:

        st.metric(
            "Absolute Error",
            f"{error:.2f}"
        )


    st.metric(
        "Percentage Error",
        f"{percentage_error:.2f}%"
    )


    st.subheader(
        f"Updated Forecast — "
        f"{new_next_date.strftime('%d %b %Y')}"
    )


    st.metric(
        "Predicted Patients",
        f"{new_prediction:.0f}"
    )


    st.info(
        f"The new observation has been added to the "
        f"training data and the model has been retrained."
    )