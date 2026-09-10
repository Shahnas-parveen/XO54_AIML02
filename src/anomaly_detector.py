import numpy as np


class FalseAlarmDetector:

    def __init__(
        self,
        baseline_window=20,
        anomaly_multiplier=3.0,
        recovery_window=2,
        persistence_window=6
    ):
        self.baseline_window = baseline_window
        self.anomaly_multiplier = anomaly_multiplier
        self.recovery_window = recovery_window
        self.persistence_window = persistence_window

        self.errors = []

        self.anomaly_active = False
        self.anomaly_start_step = None
        self.anomaly_errors = []

    def update(self, step, error):

        error = float(error)
        self.errors.append(error)

        # Build baseline
        if len(self.errors) <= self.baseline_window:

            return {
                "status": "NORMAL",
                "action": "HOLD MODEL",
                "reason": "Building baseline",
                "anomaly": False,
                "baseline_error": None,
                "current_error": error
            }

        baseline = np.mean(
            self.errors[
                -self.baseline_window - 1:-1
            ]
        )

        baseline = max(baseline, 1.0)

        # -----------------------------------
        # NORMAL → TEMPORARY ANOMALY
        # -----------------------------------

        if not self.anomaly_active:

            if error > baseline * self.anomaly_multiplier:

                self.anomaly_active = True
                self.anomaly_start_step = step
                self.anomaly_errors = [error]

                return {
                    "status": "TEMPORARY ANOMALY",
                    "action": "HOLD MODEL",
                    "reason": (
                        "Sudden deviation detected. "
                        "Holding the model while waiting "
                        "for recovery evidence."
                    ),
                    "anomaly": True,
                    "baseline_error": baseline,
                    "current_error": error
                }

            return {
                "status": "NORMAL",
                "action": "HOLD MODEL",
                "reason": "Behaviour within baseline",
                "anomaly": False,
                "baseline_error": baseline,
                "current_error": error
            }

        # -----------------------------------
        # ANOMALY ACTIVE
        # -----------------------------------

        self.anomaly_errors.append(error)

        recent = self.anomaly_errors[
            -self.recovery_window:
        ]

        # -----------------------------------
        # RECOVERY CHECK
        # -----------------------------------

        if len(recent) >= self.recovery_window:

            recent_mean = np.mean(recent)

            if recent_mean <= baseline * 1.5:

                self.anomaly_active = False

                return {
                    "status": "RECOVERED",
                    "action": "HOLD MODEL",
                    "reason": (
                        "Forecasting error returned "
                        "toward the normal baseline. "
                        "The deviation was temporary."
                    ),
                    "anomaly": False,
                    "baseline_error": baseline,
                    "current_error": error
                }

        # -----------------------------------
        # PERSISTENCE CHECK
        # -----------------------------------

        if len(self.anomaly_errors) >= self.persistence_window:

            recent_mean = np.mean(
                self.anomaly_errors[
                    -self.persistence_window:
                ]
            )

            if recent_mean > baseline * self.anomaly_multiplier:

                self.anomaly_active = False

                return {
                    "status": "CONFIRMED CHANGE",
                    "action": "ADAPT MODEL",
                    "reason": (
                        "Deviation persisted without "
                        "sufficient recovery evidence."
                    ),
                    "anomaly": False,
                    "baseline_error": baseline,
                    "current_error": error
                }

        # -----------------------------------
        # STILL WATCHING
        # -----------------------------------

        return {
            "status": "RECOVERY WATCH",
            "action": "HOLD MODEL",
            "reason": (
                "Unusual deviation detected. "
                "Waiting to determine whether "
                "the behaviour returns to normal."
            ),
            "anomaly": True,
            "baseline_error": baseline,
            "current_error": error
        }