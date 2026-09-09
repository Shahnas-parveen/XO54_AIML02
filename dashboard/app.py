import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


LOG_PATH = "logs/evaluation_log.csv"


st.set_page_config(
    page_title="Adaptive Hospital Forecasting",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏥 Adaptive Hospital Demand Forecasting")

st.caption(
    "Version 1 — Baseline Forecasting & Performance Monitoring"
)


# --------------------------------------------------
# CHECK LOG
# --------------------------------------------------

if not os.path.exists(LOG_PATH):

    st.warning(
        "No evaluation data available yet. "
        "Run run_local.py first."
    )

    st.stop()


df = pd.read_csv(LOG_PATH)


if df.empty:

    st.warning("Evaluation log is empty.")

    st.stop()


# --------------------------------------------------
# METRICS
# --------------------------------------------------

latest = df.iloc[-1]

latest_prediction = latest["predicted"]
latest_actual = latest["actual"]
latest_error = latest["absolute_error"]

mean_error = df["absolute_error"].mean()

if latest_actual != 0:
    latest_error_percent = (
        latest_error /
        abs(latest_actual)
    ) * 100
else:
    latest_error_percent = 0


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Latest Forecast",
        f"{latest_prediction:.2f}"
    )


with col2:
    st.metric(
        "Actual",
        f"{latest_actual:.2f}"
    )


with col3:
    st.metric(
        "Latest Error",
        f"{latest_error:.2f}"
    )


with col4:
    st.metric(
        "Average Error",
        f"{mean_error:.2f}"
    )


st.divider()


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.subheader("📈 Actual vs Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["step"],
        y=df["actual"],
        mode="lines+markers",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["step"],
        y=df["predicted"],
        mode="lines+markers",
        name="Forecast"
    )
)

fig.update_layout(
    xaxis_title="Time / Step",
    yaxis_title="Demand",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# ERROR
# --------------------------------------------------

st.subheader("📊 Forecast Error")

error_fig = go.Figure()

error_fig.add_trace(
    go.Scatter(
        x=df["step"],
        y=df["absolute_error"],
        mode="lines+markers",
        name="Absolute Error"
    )
)

error_fig.update_layout(
    xaxis_title="Time / Step",
    yaxis_title="Absolute Error"
)

st.plotly_chart(
    error_fig,
    use_container_width=True
)


# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.subheader("⚙️ System Status")

if latest_error_percent < 10:
    status = "🟢 NORMAL"
else:
    status = "🟡 HIGH ERROR"


col1, col2, col3 = st.columns(3)


with col1:

    st.write("### Current Status")

    st.success(status)


with col2:

    st.write("### Model")

    st.info("Baseline Forecasting Model")


with col3:

    st.write("### Adaptation")

    st.info("Not enabled in Version 1")


# --------------------------------------------------
# LATEST OBSERVATIONS
# --------------------------------------------------

st.subheader("📋 Recent Forecasts")

display_df = df.tail(10).copy()

st.dataframe(
    display_df,
    use_container_width=True
)