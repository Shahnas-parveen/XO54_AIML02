import numpy as np


class FalseAlarmDetector:

    def __init__(
        self,
        baseline_window=20,
        anomaly_multiplier=2.0,
        recovery_window=3
    ):

        self.baseline_window = baseline_window
        self.anomaly_multiplier = anomaly_multiplier
        self.recovery_window = recovery_window

        self.errors = []

        self.anomaly_active = False
        self.anomaly_start_step = None
        self.anomaly_errors = []

    def update(self, step, error):

        error = float(error)

        self.errors.append(error)

        # Build baseline first
        if len(self.errors) <= self.baseline_window:

            return {
                "status": "NORMAL",
                "action": "HOLD MODEL",
                "reason": "Building baseline",
                "anomaly": False
            }

        baseline = np.mean(
            self.errors[
                -self.baseline_window - 1:
                -1
            ]
        )

        baseline = max(baseline, 1.0)

        # ----------------------------------------
        # NEW SUDDEN DEVIATION
        # ----------------------------------------

        if not self.anomaly_active:

            if error > baseline * self.anomaly_multiplier:

                self.anomaly_active = True

                self.anomaly_start_step = step

                self.anomaly_errors = [
                    error
                ]

                return {
                    "status": "TEMPORARY ANOMALY",
                    "action": "HOLD MODEL",
                    "reason": (
                        "Sudden deviation detected. "
                        "Waiting for recovery evidence."
                    ),
                    "anomaly": True
                }

            return {
                "status": "NORMAL",
                "action": "HOLD MODEL",
                "reason": "Behaviour within baseline",
                "anomaly": False
            }

        # ----------------------------------------
        # CONTINUE WATCHING ANOMALY
        # ----------------------------------------

        self.anomaly_errors.append(error)

        recent = self.anomaly_errors[
            -self.recovery_window:
        ]

        # ----------------------------------------
        # RECOVERY
        # ----------------------------------------

        if len(recent) >= self.recovery_window:

            start_error = recent[0]

            current_error = recent[-1]

            if current_error <= start_error * 0.6:

                self.anomaly_active = False

                return {
                    "status": "RECOVERED",
                    "action": "HOLD MODEL",
                    "reason": (
                        "Error returned toward the "
                        "normal range. Temporary "
                        "deviation confirmed."
                    ),
                    "anomaly": False
                }

        # ----------------------------------------
        # PERSISTENT CHANGE
        # ----------------------------------------

        if len(self.anomaly_errors) >= self.recovery_window:

            recent_mean = np.mean(
                self.anomaly_errors[
                    -self.recovery_window:
                ]
            )

            if recent_mean > (
                baseline *
                self.anomaly_multiplier
            ):

                self.anomaly_active = False

                return {
                    "status": "CONFIRMED CHANGE",
                    "action": "ADAPT MODEL",
                    "reason": (
                        "Deviation persisted without "
                        "sufficient recovery evidence."
                    ),
                    "anomaly": False
                }

        return {
            "status": "RECOVERY WATCH",
            "action": "HOLD MODEL",
            "reason": (
                "Unusual deviation detected. "
                "Waiting for recovery."
            ),
            "anomaly": True
        }