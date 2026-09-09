import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

SC1_LOG = "logs/sc1_log.csv"
SC2_LOG = "logs/sc2_log.csv"

st.set_page_config(
    page_title="Adaptive Hospital Demand Forecasting",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_log(path):
    """Safely load an evaluation log."""

    if not os.path.exists(path):
        return pd.DataFrame()

    if os.path.getsize(path) == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    return df


def get_error_column(df):

    if "absolute_error" in df.columns:
        return "absolute_error"

    if "error" in df.columns:
        return "error"

    return None


def show_common_metrics(df):

    if df.empty:
        st.info("No evaluation data available yet.")
        return

    error_column = get_error_column(df)

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "predicted" in df.columns:
            st.metric(
                "Latest Forecast",
                f"{latest['predicted']:.2f}"
            )

    with col2:
        if "actual" in df.columns:
            st.metric(
                "Latest Actual",
                f"{latest['actual']:.2f}"
            )

    with col3:
        if error_column:
            st.metric(
                "Latest Error",
                f"{latest[error_column]:.2f}"
            )

    with col4:
        if error_column:
            st.metric(
                "Average Error",
                f"{df[error_column].mean():.2f}"
            )


def show_forecast_chart(df, title):

    if "timestamp" not in df.columns:
        x_axis = df["step"]
        x_title = "Step"
    else:
        x_axis = df["timestamp"]
        x_title = "Date"

    if "actual" not in df.columns or "predicted" not in df.columns:
        return

    st.subheader("📈 Actual vs Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df["actual"],
            mode="lines+markers",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df["predicted"],
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Target / Demand",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def show_error_chart(df):

    error_column = get_error_column(df)

    if error_column is None:
        return

    if "timestamp" in df.columns:
        x_axis = df["timestamp"]
        x_title = "Date"
    else:
        x_axis = df["step"]
        x_title = "Step"

    st.subheader("📊 Forecast Error")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df[error_column],
            mode="lines+markers",
            name="Absolute Error"
        )
    )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Absolute Error",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def show_detection(df, challenge):

    st.subheader("🔍 Change Detection Monitor")

    if df.empty:
        st.info(
            f"No {challenge} evaluation data available yet."
        )
        return

    latest = df.iloc[-1]

    # --------------------------------------------------------
    # Determine status
    # --------------------------------------------------------

    status = str(
        latest.get("status", "NORMAL")
    )

    if challenge == "SC1":

        if "change_score" in df.columns:
            change_score = latest["change_score"]
        else:
            change_score = 0.0

        if "baseline_error" in df.columns:
            baseline_error = latest["baseline_error"]
        else:
            baseline_error = None

        if "recent_error" in df.columns:
            recent_error = latest["recent_error"]
        else:
            recent_error = None

        if "error_trend" in df.columns:
            trend = latest["error_trend"]
        else:
            trend = None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current State",
                status
            )

        with col2:
            st.metric(
                "Change Score",
                f"{change_score:.3f}"
            )

        with col3:
            if baseline_error is not None:
                st.metric(
                    "Baseline Error",
                    f"{baseline_error:.2f}"
                )
            else:
                st.metric(
                    "Baseline Error",
                    "N/A"
                )

        with col4:
            if recent_error is not None:
                st.metric(
                    "Recent Error",
                    f"{recent_error:.2f}"
                )
            else:
                st.metric(
                    "Recent Error",
                    "N/A"
                )

        st.write("### 📌 Detection Evidence")

        if trend is not None:
            st.write(
                f"**Error Trend:** `{trend:.4f}`"
            )

        reason = latest.get(
            "reason",
            "No explanation recorded."
        )

        st.info(
            f"**Evidence:** {reason}"
        )

        action = latest.get(
            "adaptation",
            "HOLD MODEL"
        )

        st.write(
            f"**Model Action:** `{action}`"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Current State",
                status
            )

        with col2:

            action = latest.get(
                "adaptation",
                latest.get(
                    "action",
                    "HOLD MODEL"
                )
            )

            st.metric(
                "Model Action",
                str(action)
            )

        st.write("### 📌 Detection Evidence")

        reason = latest.get(
            "reason",
            "No explanation recorded."
        )

        st.info(
            f"**Evidence:** {reason}"
        )

    # --------------------------------------------------------
    # Status over time
    # --------------------------------------------------------

    if (
        "timestamp" in df.columns
        and "status" in df.columns
    ):

        st.subheader("📊 Behaviour Status Over Time")

        status_map = {
            "NORMAL": 0,
            "WATCH": 1,
            "TEMPORARY ANOMALY": 1,
            "RECOVERY WATCH": 1,
            "RECOVERED": 0,
            "CONFIRMED CHANGE": 2
        }

        plot_df = df.copy()

        plot_df["status_value"] = (
            plot_df["status"]
            .map(status_map)
            .fillna(0)
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["status_value"],
                mode="lines+markers",
                name="Detection State",
                text=plot_df["status"],
                hovertemplate=(
                    "Date: %{x}<br>"
                    "State: %{text}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Detection State",
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=[
                    "NORMAL",
                    "WATCH / ANOMALY",
                    "CONFIRMED CHANGE"
                ]
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


def show_recent_data(df):

    st.subheader("📋 Recent Evaluation")

    if df.empty:
        st.info("No data available.")
        return

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏥 Adaptive Hospital Demand Forecasting"
)

st.caption(
    "Adaptive time-series forecasting with behavioural "
    "change detection and false-alarm handling"
)


st.divider()


# ============================================================
# LOAD DATA
# ============================================================

sc1_df = load_log(SC1_LOG)
sc2_df = load_log(SC2_LOG)


# ============================================================
# OVERALL STATUS
# ============================================================

st.subheader("🧠 Evaluation Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.write("### Model")

    st.info(
        "Random Forest Forecasting Model"
    )

with col2:

    st.write("### Challenge 1")

    if sc1_df.empty:
        st.warning("Not evaluated yet")
    else:
        latest_status = str(
            sc1_df.iloc[-1].get(
                "status",
                "NORMAL"
            )
        )

        st.success(
            f"SC1: {latest_status}"
        )

with col3:

    st.write("### Challenge 2")

    if sc2_df.empty:
        st.warning("Not evaluated yet")
    else:
        latest_status = str(
            sc2_df.iloc[-1].get(
                "status",
                "NORMAL"
            )
        )

        st.success(
            f"SC2: {latest_status}"
        )


st.divider()


# ============================================================
# CHALLENGE TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔵 Challenge 1 — Quiet Shift",
        "🟣 Challenge 2 — False Alarm"
    ]
)


# ============================================================
# SC1
# ============================================================

with tab1:

    st.header(
        "🔵 Surprise Challenge 1 — The Quiet Shift"
    )

    st.caption(
        "Detect gradual behavioural change using "
        "sustained evidence rather than a single unusual error."
    )

    if sc1_df.empty:

        st.warning(
            "SC1 data is not available yet. "
            "Complete the SC1 API evaluation to populate this section."
        )

    else:

        show_common_metrics(
            sc1_df
        )

        st.divider()

        show_forecast_chart(
            sc1_df,
            "SC1 — Actual vs Forecast"
        )

        show_error_chart(
            sc1_df
        )

        show_detection(
            sc1_df,
            "SC1"
        )

        show_recent_data(
            sc1_df
        )


# ============================================================
# SC2
# ============================================================

with tab2:

    st.header(
        "🟣 Surprise Challenge 2 — The False Alarm"
    )

    st.caption(
        "Detect sudden deviations while avoiding unnecessary "
        "model adaptation when the behaviour returns to normal."
    )

    if sc2_df.empty:

        st.warning(
            "SC2 data is not available yet. "
            "Complete the SC2 API evaluation to populate this section."
        )

    else:

        show_common_metrics(
            sc2_df
        )

        st.divider()

        show_forecast_chart(
            sc2_df,
            "SC2 — Actual vs Forecast"
        )

        show_error_chart(
            sc2_df
        )

        show_detection(
            sc2_df,
            "SC2"
        )

        show_recent_data(
            sc2_df
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write(
        "### Data Sources"
    )

    st.write(
        "🔵 SC1 API → `sc1_log.csv`"
    )

    st.write(
        "🟣 SC2 API → `sc2_log.csv`"
    )

    st.divider()

    st.write(
        "### Decision Logic"
    )

    st.write(
        "**SC1:**\n"
        "NORMAL → WATCH → CONFIRMED CHANGE"
    )

    st.write(
        "**SC2:**\n"
        "NORMAL → TEMPORARY ANOMALY → "
        "RECOVERED / CONFIRMED CHANGE"
    )

    st.divider()

    st.caption(
        "Predictions are generated sequentially. "
        "Actual values are revealed by the evaluation API "
        "after each prediction."
    )