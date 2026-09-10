import argparse
import json
import os

import pandas as pd

from src.api_client import EvaluationAPI
from src.model import ForecastingModel
from src.features import build_live_features
from src.features import get_feature_columns
from src.change_detector import QuietShiftDetector
from src.anomaly_detector import FalseAlarmDetector


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# SESSION MANAGEMENT
# ============================================================

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


# ============================================================
# MODEL ADAPTATION
# ============================================================

def adapt_model(model, observed_data):

    if len(observed_data) < 10:

        print(
            ">>> Not enough observations for adaptation."
        )

        return False

    recent_window = 50

    recent_data = (
        observed_data
        .tail(recent_window)
        .copy()
    )

    # Create the same engineered features
    # used during initial training.
    from src.features import add_features

    engineered = add_features(
        recent_data
    )

    feature_columns = get_feature_columns()

    engineered = engineered.dropna()

    if len(engineered) < 5:

        print(
            ">>> Not enough valid recent data "
            "for adaptation."
        )

        return False

    X = engineered[
        feature_columns
    ]

    y = engineered[
        "target"
    ]

    model.train(
        X,
        y
    )

    print(
        f">>> MODEL ADAPTED using "
        f"{len(engineered)} recent observations."
    )

    return True


# ============================================================
# LIVE STREAM
# ============================================================

def run_stream(stream):

    print("=" * 60)
    print(
        f"STARTING LIVE EVALUATION: "
        f"{stream.upper()}"
    )
    print("=" * 60)


    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    existing_session = (
        load_existing_session(stream)
    )

    if existing_session:

        print(
            f"Existing {stream.upper()} session found:"
        )

        print(
            existing_session
        )

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

        session_id = session[
            "session_id"
        ]

        save_session(
            stream,
            session_id
        )

        print(
            "New session created:"
        )

        print(
            session_id
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = ForecastingModel()

    model.load(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # HISTORICAL TARGET HISTORY
    # --------------------------------------------------------

    training_df = pd.read_csv(
        TRAINING_PATH
    )

    training_df[
        "timestamp"
    ] = pd.to_datetime(
        training_df["timestamp"]
    )

    target_history = (
        training_df[
            "target"
        ]
        .astype(float)
        .tolist()
    )


    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    if stream == "sc1":

        detector = QuietShiftDetector()

    else:

        detector = FalseAlarmDetector()


    # --------------------------------------------------------
    # LIVE OBSERVATION HISTORY
    # --------------------------------------------------------

    observed_data = []

    logs = []

    os.makedirs(
        "logs",
        exist_ok=True
    )


    # ========================================================
    # API LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # 1. GET NEXT ROW
        # ----------------------------------------------------

        nxt = api.get_next()


        if nxt.get("done"):

            print()
            print(
                "=" * 60
            )

            print(
                f"{stream.upper()} "
                f"COMPLETED SUCCESSFULLY."
            )

            print(
                "=" * 60
            )

            break


        step = nxt[
            "step"
        ]

        timestamp = nxt[
            "timestamp"
        ]


        # ----------------------------------------------------
        # 2. GET FEATURES
        # ----------------------------------------------------

        features = {
            column: nxt[column]
            for column in FEATURE_COLUMNS
        }


        # ----------------------------------------------------
        # 3. BUILD LIVE FEATURES
        # ----------------------------------------------------

        X_live = build_live_features(
            timestamp,
            features,
            target_history
        )


        # ----------------------------------------------------
        # 4. PREDICT
        # ----------------------------------------------------

        prediction = float(
            model.predict(
                X_live
            )[0]
        )


        # ----------------------------------------------------
        # 5. SUBMIT PREDICTION
        # ----------------------------------------------------

        result = api.submit_prediction(
            prediction
        )


        # ----------------------------------------------------
        # 6. RECEIVE ACTUAL
        # ----------------------------------------------------

        actual = float(
            result[
                "actual_target"
            ]
        )


        # ----------------------------------------------------
        # 7. CALCULATE ERROR
        # ----------------------------------------------------

        signed_error = (
            actual - prediction
        )

        absolute_error = abs(
            signed_error
        )

        if actual != 0:

            percentage_error = (
                absolute_error
                / abs(actual)
            ) * 100

        else:

            percentage_error = 0.0


        # ----------------------------------------------------
        # 8. UPDATE TARGET HISTORY
        # ----------------------------------------------------

        target_history.append(
            actual
        )


        # ----------------------------------------------------
        # 9. SAVE OBSERVED DATA
        # ----------------------------------------------------

        observed_data.append(
            {
                "timestamp": timestamp,
                **features,
                "target": actual
            }
        )


        # ----------------------------------------------------
        # 10. CHANGE / ANOMALY DETECTION
        # ----------------------------------------------------

        decision = detector.update(
            step,
            absolute_error
        )


        # ----------------------------------------------------
        # 11. MODEL ACTION
        # ----------------------------------------------------

        adaptation = (
            "HOLD MODEL"
        )


        if (
            decision["status"]
            == "CONFIRMED CHANGE"
        ):

            print()
            print(
                ">>> CONFIRMED CHANGE DETECTED"
            )

            adapted = adapt_model(
                model,
                pd.DataFrame(
                    observed_data
                )
            )

            if adapted:

                adaptation = (
                    "MODEL ADAPTED"
                )

            else:

                adaptation = (
                    "ADAPTATION SKIPPED"
                )


        # ----------------------------------------------------
        # 12. CREATE LOG RECORD
        # ----------------------------------------------------

        record = {

            "step":
                step,

            "timestamp":
                timestamp,

            "predicted":
                prediction,

            "actual":
                actual,

            "error":
                signed_error,

            "absolute_error":
                absolute_error,

            "percentage_error":
                percentage_error,

            "status":
                decision[
                    "status"
                ],

            "adaptation":
                adaptation,

            "reason":
                decision[
                    "reason"
                ]
        }


        # ----------------------------------------------------
        # SC1 INFORMATION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SC2 INFORMATION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 13. SAVE LOG AFTER EVERY OBSERVATION
        # ----------------------------------------------------

        logs.append(
            record
        )

        pd.DataFrame(
            logs
        ).to_csv(
            f"logs/{stream}_log.csv",
            index=False
        )


        # ----------------------------------------------------
        # 14. DISPLAY STATUS
        # ----------------------------------------------------

        print(
            f"Step {step:03d} | "
            f"Pred={prediction:.2f} | "
            f"Actual={actual:.2f} | "
            f"Error={absolute_error:.2f} | "
            f"{decision['status']} | "
            f"{adaptation}"
        )


    print()

    print(
        f"Log saved to "
        f"logs/{stream}_log.csv"
    )

    return session_id


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stream",
        choices=[
            "sc1",
            "sc2"
        ]
    )

    args = parser.parse_args()

    run_stream(
        args.stream
    )