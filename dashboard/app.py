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
    "Challenge 1 — The Quiet Shift | "
    "Sustained Behaviour Change Detection"
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


try:

    df = pd.read_csv(LOG_PATH)

except pd.errors.EmptyDataError:

    st.warning(
        "Evaluation log is empty. "
        "Run run_local.py first."
    )

    st.stop()


if df.empty:

    st.warning("Evaluation log is empty.")

    st.stop()


# --------------------------------------------------
# PREPARE TIMESTAMP
# --------------------------------------------------

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )


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


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

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
# ACTUAL VS FORECAST
# --------------------------------------------------

st.subheader("📈 Actual vs Forecast")


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["actual"],
        mode="lines+markers",
        name="Actual"
    )
)


fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["predicted"],
        mode="lines+markers",
        name="Forecast"
    )
)


fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Demand",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# FORECAST ERROR
# --------------------------------------------------

st.subheader("📊 Forecast Error")


error_fig = go.Figure()


error_fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["absolute_error"],
        mode="lines+markers",
        name="Absolute Error"
    )
)


error_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Absolute Error",
    hovermode="x unified"
)


st.plotly_chart(
    error_fig,
    use_container_width=True
)


# --------------------------------------------------
# QUIET SHIFT MONITOR
# --------------------------------------------------

st.subheader("🔍 Quiet Shift Monitor")


latest_status = str(
    latest.get(
        "status",
        "NORMAL"
    )
)


latest_score = latest.get(
    "change_score",
    0
)


baseline_error = latest.get(
    "baseline_error",
    None
)


recent_error = latest.get(
    "recent_error",
    None
)


error_trend = latest.get(
    "error_trend",
    0
)


latest_reason = str(
    latest.get(
        "reason",
        "No additional information"
    )
)


# --------------------------------------------------
# CURRENT STATE
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.write("### Current State")


    if latest_status == "NORMAL":

        st.success("🟢 NORMAL")


    elif latest_status == "WATCH":

        st.warning("🟡 WATCH")


    elif latest_status == "CONFIRMED CHANGE":

        st.error("🔴 CONFIRMED CHANGE")


    else:

        st.info(latest_status)


with col2:

    st.write("### Change Score")

    if pd.notna(latest_score):

        st.metric(
            "Score",
            f"{float(latest_score):.3f}"
        )

    else:

        st.metric(
            "Score",
            "Building"
        )


with col3:

    st.write("### Model Action")


    if latest_status == "CONFIRMED CHANGE":

        st.error("ADAPT MODEL")

    else:

        st.info("HOLD MODEL")


# --------------------------------------------------
# CHANGE EVIDENCE
# --------------------------------------------------

st.subheader("🧠 Change Detection Evidence")


col1, col2, col3 = st.columns(3)


with col1:

    st.write("**Baseline Error**")

    if (
        baseline_error is not None
        and pd.notna(baseline_error)
    ):

        st.write(
            f"{float(baseline_error):.2f}"
        )

    else:

        st.write(
            "Building baseline"
        )


with col2:

    st.write("**Recent Rolling Error**")

    if (
        recent_error is not None
        and pd.notna(recent_error)
    ):

        st.write(
            f"{float(recent_error):.2f}"
        )

    else:

        st.write(
            "Building window"
        )


with col3:

    st.write("**Error Trend**")

    if pd.notna(error_trend):

        st.write(
            f"{float(error_trend):.4f}"
        )

    else:

        st.write("Building trend")


st.info(
    f"**Evidence:** {latest_reason}"
)


# --------------------------------------------------
# STATUS OVER TIME
# --------------------------------------------------

st.subheader("📊 Behaviour Status Over Time")


if "status" in df.columns:

    status_df = df[
        [
            "timestamp",
            "status"
        ]
    ].copy()


    status_map = {

        "NORMAL": 0,

        "WATCH": 1,

        "CONFIRMED CHANGE": 2

    }


    status_df["status_value"] = (
        status_df["status"]
        .map(status_map)
        .fillna(0)
    )


    status_fig = go.Figure()


    status_fig.add_trace(
        go.Scatter(
            x=status_df["timestamp"],
            y=status_df["status_value"],
            mode="lines+markers",
            name="Detection State"
        )
    )


    status_fig.update_layout(

        xaxis_title="Date",

        yaxis_title="Detection State",

        yaxis=dict(
            tickmode="array",
            tickvals=[
                0,
                1,
                2
            ],
            ticktext=[
                "NORMAL",
                "WATCH",
                "CONFIRMED CHANGE"
            ]
        ),

        hovermode="x unified"
    )


    st.plotly_chart(
        status_fig,
        use_container_width=True
    )

# --------------------------------------------------
# DETECTION SUMMARY
# --------------------------------------------------

st.subheader("💡 Detection Summary")

st.caption(
    "The system uses rolling error, error trend, and sustained "
    "evidence to distinguish normal behaviour from a genuine shift."
)

# --------------------------------------------------
# RECENT FORECASTS
# --------------------------------------------------

st.subheader("📋 Recent Forecasts")


display_columns = [

    "step",

    "timestamp",

    "predicted",

    "actual",

    "absolute_error",

    "status",

    "adaptation",

    "change_score",

    "baseline_error",

    "recent_error",

    "error_trend"

]


available_columns = [

    column

    for column in display_columns

    if column in df.columns

]


display_df = df.tail(10)[
    available_columns
].copy()


st.dataframe(
    display_df,
    use_container_width=True
)