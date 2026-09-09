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


st.set_page_config(
    page_title="Adaptive Hospital Demand Forecasting",
    layout="wide"
)


st.title("🏥 Adaptive Hospital Demand Forecasting")

st.write(
    "Predict next-day patient demand and evaluate "
    "the prediction when the actual count becomes available."
)


# ============================================
# STEP 1 — HISTORICAL DATA
# ============================================

st.header("1. Historical Patient Data")

uploaded_file = st.file_uploader(
    "Upload training data",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "timestamp" not in df.columns:
        st.error("CSV must contain a timestamp column.")
        st.stop()

    if "target" not in df.columns:
        st.error("CSV must contain a target column.")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    st.success(
        f"Loaded {len(df)} historical observations."
    )


    # ============================================
    # STEP 2 — TRAIN MODEL
    # ============================================

    model = PatientForecaster()

    model.train(df)

    st.success("Forecasting model trained successfully.")


    # ============================================
    # STEP 3 — NEXT DAY PREDICTION
    # ============================================

    prediction = model.predict_next(df)

    last_date = df["timestamp"].iloc[-1]

    next_date = last_date + pd.Timedelta(days=1)


    st.header("2. Next-Day Patient Demand")

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


    # ============================================
    # STEP 4 — ACTUAL VALUE
    # ============================================

    st.header("3. Enter Actual Patient Count")

    actual_value = st.number_input(
        f"Actual patient count for "
        f"{next_date.strftime('%d %b %Y')}",
        min_value=0.0,
        step=1.0
    )


    # ============================================
    # STEP 5 — CALCULATE ERROR
    # ============================================

    if st.button("Evaluate Forecast"):

        error = abs(
            actual_value - prediction
        )

        percentage_error = (
            error / actual_value * 100
            if actual_value != 0
            else 0
        )


        st.header("4. Forecast Evaluation")

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


        # ============================================
        # STEP 6 — RETRAIN
        # ============================================

        st.header("5. Model Adaptation")

        updated_row = pd.DataFrame({
            "timestamp": [next_date],
            "target": [actual_value]
        })


        updated_df = pd.concat(
            [
                df[["timestamp", "target"]],
                updated_row
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


        if st.button("Retrain Model"):

            model.train(updated_df)

            new_prediction = model.predict_next(
                updated_df
            )

            new_date = (
                updated_df["timestamp"].iloc[-1]
                + pd.Timedelta(days=1)
            )


            st.success(
                "Model retrained using the new patient count."
            )


            st.metric(
                f"New Prediction — {new_date.strftime('%d %b %Y')}",
                f"{new_prediction:.0f} patients"
            )