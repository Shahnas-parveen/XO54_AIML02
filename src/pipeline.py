import os

import pandas as pd

from src.monitor import calculate_error
from src.change_detector import QuietShiftDetector


LOG_PATH = "logs/evaluation_log.csv"


class ForecastPipeline:

    def __init__(self, model):

        self.model = model

        self.history = []
        self.error_history = []

        # Challenge 1: Quiet Shift Detector
        self.change_detector = QuietShiftDetector()

        os.makedirs(
            "logs",
            exist_ok=True
        )

    def predict(self, features):

        prediction = self.model.predict(
            features
        )[0]

        return float(prediction)

    def update(
        self,
        step,
        timestamp,
        prediction,
        actual
    ):

        # --------------------------------------------------
        # CALCULATE ERROR
        # --------------------------------------------------

        metrics = calculate_error(
            actual,
            prediction
        )

        absolute_error = (
            metrics["absolute_error"]
        )

        self.error_history.append(
            absolute_error
        )

        # --------------------------------------------------
        # CHALLENGE 1
        # QUIET SHIFT DETECTION
        # --------------------------------------------------

        change_result = (
            self.change_detector.update(
                step,
                absolute_error
            )
        )

        # --------------------------------------------------
        # MODEL ACTION
        # --------------------------------------------------

        if change_result["status"] == "CONFIRMED CHANGE":

            adaptation = "ADAPT MODEL"

        elif change_result["status"] == "WATCH":

            adaptation = "HOLD MODEL"

        else:

            adaptation = "HOLD MODEL"

        # --------------------------------------------------
        # CREATE LOG RECORD
        # --------------------------------------------------

        record = {

            "step": step,

            "timestamp": timestamp,

            "predicted": prediction,

            "actual": actual,

            "error": metrics["error"],

            "absolute_error": absolute_error,

            "percentage_error":
                metrics["percentage_error"],

            "status":
                change_result["status"],

            "adaptation":
                adaptation,

            "change_score":
                change_result["change_score"],

            "baseline_error":
                change_result["baseline_error"],

            "recent_error":
                change_result["recent_error"],

            "error_trend":
                change_result["trend"],

            "reason":
                change_result["reason"]
        }

        self.history.append(
            record
        )

        self._save_log()

        return record

    def _save_log(self):

        df = pd.DataFrame(
            self.history
        )

        df.to_csv(
            LOG_PATH,
            index=False
        )

    def get_history(self):

        return pd.DataFrame(
            self.history
        )