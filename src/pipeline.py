import os
import pandas as pd

from src.monitor import calculate_error


LOG_PATH = "logs/evaluation_log.csv"


class ForecastPipeline:

    def __init__(self, model):

        self.model = model

        self.history = []
        self.error_history = []

        os.makedirs("logs", exist_ok=True)

    def predict(self, features):

        prediction = self.model.predict(features)[0]

        return float(prediction)

    def update(self, step, timestamp, prediction, actual):

        metrics = calculate_error(
            actual,
            prediction
        )

        self.error_history.append(
            metrics["absolute_error"]
        )

        record = {
            "step": step,
            "timestamp": timestamp,
            "predicted": prediction,
            "actual": actual,
            "error": metrics["error"],
            "absolute_error": metrics["absolute_error"],
            "percentage_error": metrics["percentage_error"],
            "status": "NORMAL",
            "adaptation": "NONE"
        }

        self.history.append(record)

        self._save_log()

        return record

    def _save_log(self):

        df = pd.DataFrame(self.history)

        df.to_csv(
            LOG_PATH,
            index=False
        )

    def get_history(self):

        return pd.DataFrame(self.history)