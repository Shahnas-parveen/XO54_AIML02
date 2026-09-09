import argparse
import json
import os

import pandas as pd

from src.api_client import EvaluationAPI
from src.model import ForecastingModel
from src.features import build_live_features
from src.change_detector import QuietShiftDetector
from src.anomaly_detector import FalseAlarmDetector


TEAM_NAME = "Neural Ninjas"

MODEL_PATH = "models/forecasting_model.pkl"
TRAINING_PATH = "data/training_set.csv"
SESSION_DIR = "sessions"


FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6"
]


def save_session(stream, session_id):

    os.makedirs(
        SESSION_DIR,
        exist_ok=True
    )

    path = os.path.join(
        SESSION_DIR,
        f"{stream}_session.json"
    )

    with open(path, "w") as f:
        json.dump(
            {
                "stream": stream,
                "session_id": session_id
            },
            f,
            indent=4
        )


def load_existing_session(stream):

    path = os.path.join(
        SESSION_DIR,
        f"{stream}_session.json"
    )

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return data["session_id"]


def run_stream(stream):

    print("=" * 60)
    print(f"STARTING LIVE EVALUATION: {stream.upper()}")
    print("=" * 60)

    existing_session = load_existing_session(stream)

    if existing_session:

        print(
            f"Existing {stream.upper()} session found:"
        )
        print(existing_session)

        session_id = existing_session

        api = EvaluationAPI(
            team_name=TEAM_NAME,
            stream=stream,
            session_id=session_id
        )

    else:

        api = EvaluationAPI(
            team_name=TEAM_NAME,
            stream=stream
        )

        session = api.start_session()

        session_id = session["session_id"]

        save_session(
            stream,
            session_id
        )

        print("New session created:")
        print(session_id)


    # Load improved model
    model = ForecastingModel()

    model.load(
        MODEL_PATH
    )


    # Seed target history using historical training data
    training_df = pd.read_csv(
        TRAINING_PATH
    )

    target_history = (
        training_df["target"]
        .astype(float)
        .tolist()
    )


    if stream == "sc1":

        detector = QuietShiftDetector()

    else:

        detector = FalseAlarmDetector()


    logs = []

    os.makedirs(
        "logs",
        exist_ok=True
    )


    while True:

        nxt = api.get_next()

        if nxt.get("done"):

            print()
            print(
                f"{stream.upper()} COMPLETED SUCCESSFULLY."
            )

            break


        step = nxt["step"]

        timestamp = nxt["timestamp"]


        features = {
            column: nxt[column]
            for column in FEATURE_COLUMNS
        }


        # Build features using previous observed targets
        X_live = build_live_features(
            timestamp,
            features,
            target_history
        )


        prediction = float(
            model.predict(X_live)[0]
        )


        # IMPORTANT:
        # Prediction is submitted BEFORE seeing actual
        result = api.submit_prediction(
            prediction
        )


        actual = float(
            result["actual_target"]
        )


        error = abs(
            actual - prediction
        )


        # Now actual is known and becomes history
        target_history.append(
            actual
        )


        decision = detector.update(
            step,
            error
        )


        adaptation = "HOLD MODEL"


        # For a confirmed lasting change,
        # rebuild model using recent observations.
        if decision["status"] == "CONFIRMED CHANGE":

            adaptation = "ADAPT MODEL"

            print(
                ">>> CONFIRMED CHANGE DETECTED"
            )


        record = {
            "step": step,
            "timestamp": timestamp,
            "predicted": prediction,
            "actual": actual,
            "absolute_error": error,
            "status": decision["status"],
            "adaptation": adaptation,
            "reason": decision["reason"]
        }


        if stream == "sc1":

            record.update({
                "change_score":
                    decision.get(
                        "change_score"
                    ),
                "baseline_error":
                    decision.get(
                        "baseline_error"
                    ),
                "recent_error":
                    decision.get(
                        "recent_error"
                    ),
                "error_trend":
                    decision.get(
                        "trend"
                    ),
                "confirmed_step":
                    decision.get(
                        "confirmed_step"
                    )
            })

        else:

            record.update({
                "baseline_error":
                    decision.get(
                        "baseline_error"
                    ),
                "current_error":
                    decision.get(
                        "current_error"
                    ),
                "anomaly":
                    decision.get(
                        "anomaly"
                    )
            })


        logs.append(record)


        # Save after EVERY API observation
        pd.DataFrame(logs).to_csv(
            f"logs/{stream}_log.csv",
            index=False
        )


        print(
            f"Step {step:03d} | "
            f"Pred={prediction:.2f} | "
            f"Actual={actual:.2f} | "
            f"Error={error:.2f} | "
            f"{decision['status']} | "
            f"{adaptation}"
        )


    print()
    print(
        f"Log saved to logs/{stream}_log.csv"
    )

    return session_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stream",
        choices=["sc1", "sc2"]
    )

    args = parser.parse_args()

    run_stream(
        args.stream
    )