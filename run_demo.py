import os
import pandas as pd
import numpy as np

from src.model import ForecastingModel
from src.features import build_live_features
from src.features import add_features, get_feature_columns
from src.change_detector import QuietShiftDetector
from src.anomaly_detector import FalseAlarmDetector


TRAINING_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6"
]


def adapt_model(model, observed_data):
    if len(observed_data) < 15:
        return False

    recent_data = observed_data.tail(50).copy()

    engineered = add_features(recent_data)
    engineered = engineered.dropna()

    if len(engineered) < 5:
        return False

    feature_columns = get_feature_columns()

    X = engineered[feature_columns]
    y = engineered["target"]

    model.train(X, y)

    return True


def create_demo_data():
    """
    Create two local evaluation streams using the real training data.

    SC1:
    Gradual sustained increase in target behaviour.

    SC2:
    Sudden temporary spike followed by recovery.
    """

    df = pd.read_csv(TRAINING_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Use the final 60 historical observations
    demo = df.tail(60).copy().reset_index(drop=True)

    # -----------------------------
    # SC1 - QUIET SHIFT
    # -----------------------------
    sc1 = demo.copy()

    # Gradually introduce a shift
    shift_start = 35

    for i in range(shift_start, len(sc1)):
        progress = (i - shift_start) / (len(sc1) - shift_start - 1)

        # gradual increase
        shift = 3 + (progress * 14)

        sc1.loc[i, "target"] = sc1.loc[i, "target"] + shift

    # -----------------------------
    # SC2 - FALSE ALARM
    # -----------------------------
    sc2 = demo.copy()

    anomaly_start = 35
    anomaly_end = 40

    for i in range(anomaly_start, anomaly_end):
        # sudden temporary deviation
        sc2.loc[i, "target"] = sc2.loc[i, "target"] + 25

    # After anomaly, target naturally returns
    # to the original training behaviour.

    os.makedirs("demo_data", exist_ok=True)

    sc1.to_csv("demo_data/sc1_demo.csv", index=False)
    sc2.to_csv("demo_data/sc2_demo.csv", index=False)

    print("Demo datasets created:")
    print("  demo_data/sc1_demo.csv")
    print("  demo_data/sc2_demo.csv")

    return sc1, sc2


def run_demo(stream, demo_df):

    print("\n" + "=" * 70)
    print(f"RUNNING LOCAL DEMO: {stream.upper()}")
    print("=" * 70)

    model = ForecastingModel()
    model.load(MODEL_PATH)

    if stream == "sc1":
        detector = QuietShiftDetector()
    else:
        detector = FalseAlarmDetector()

    # Seed history with complete historical training targets
    training_df = pd.read_csv(TRAINING_PATH)
    target_history = training_df["target"].astype(float).tolist()

    observed_data = []
    logs = []

    adaptation_done = False

    for i in range(len(demo_df)):

        row = demo_df.iloc[i]

        timestamp = row["timestamp"]

        features = {
            column: float(row[column])
            for column in FEATURE_COLUMNS
        }

        # Build exactly the same live features
        X_live = build_live_features(
            timestamp,
            features,
            target_history
        )

        # Forecast
        prediction = float(model.predict(X_live)[0])

        # Actual target from simulated stream
        actual = float(row["target"])

        error = actual - prediction
        absolute_error = abs(error)

        percentage_error = (
            absolute_error / abs(actual) * 100
            if actual != 0
            else 0
        )

        # IMPORTANT:
        # Actual becomes available after prediction,
        # just like the real API.
        target_history.append(actual)

        observed_data.append({
            "timestamp": timestamp,
            **features,
            "target": actual
        })

        # Detector sees forecasting error
        decision = detector.update(
            i,
            absolute_error
        )

        adaptation = "HOLD MODEL"

        # Adapt only after confirmed change
        if decision["status"] == "CONFIRMED CHANGE" and not adaptation_done:

            adapted = adapt_model(
                model,
                pd.DataFrame(observed_data)
            )

            if adapted:
                adaptation = "MODEL ADAPTED"
                adaptation_done = True

        record = {
            "step": i,
            "timestamp": timestamp,
            "predicted": prediction,
            "actual": actual,
            "error": error,
            "absolute_error": absolute_error,
            "percentage_error": percentage_error,
            "status": decision["status"],
            "adaptation": adaptation,
            "reason": decision["reason"]
        }

        if stream == "sc1":

            record.update({
                "change_score":
                    decision.get("change_score"),

                "baseline_error":
                    decision.get("baseline_error"),

                "recent_error":
                    decision.get("recent_error"),

                "error_trend":
                    decision.get("trend"),

                "confirmed_step":
                    decision.get("confirmed_step")
            })

        else:

            record.update({
                "baseline_error":
                    decision.get("baseline_error"),

                "current_error":
                    decision.get("current_error"),

                "anomaly":
                    decision.get("anomaly")
            })

        logs.append(record)

        print(
            f"Step {i:02d} | "
            f"Pred={prediction:6.2f} | "
            f"Actual={actual:6.2f} | "
            f"Error={absolute_error:6.2f} | "
            f"{decision['status']:18s} | "
            f"{adaptation}"
        )

    os.makedirs("logs", exist_ok=True)

    output = f"logs/{stream}_log.csv"

    pd.DataFrame(logs).to_csv(
        output,
        index=False
    )

    print("\n" + "-" * 70)
    print(f"{stream.upper()} DEMO COMPLETED")
    print(f"Log saved to: {output}")

    # Summary
    log_df = pd.DataFrame(logs)

    print("\nSummary:")
    print(
        "Average Error:",
        round(log_df["absolute_error"].mean(), 2)
    )

    print(
        "Max Error:",
        round(log_df["absolute_error"].max(), 2)
    )

    print("\nStatus counts:")
    print(log_df["status"].value_counts())

    print("=" * 70)


def main():

    print("=" * 70)
    print("ADAPTIVE HOSPITAL DEMAND FORECASTING")
    print("LOCAL DEMONSTRATION MODE")
    print("=" * 70)

    # Create simulated streams
    sc1, sc2 = create_demo_data()

    # Run SC1
    run_demo(
        "sc1",
        sc1
    )

    # Run SC2
    run_demo(
        "sc2",
        sc2
    )

    print("\n")
    print("=" * 70)
    print("ALL LOCAL DEMOS COMPLETED")
    print("=" * 70)
    print("\nOpen dashboard using:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()