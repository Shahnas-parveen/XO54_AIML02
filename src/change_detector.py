import numpy as np


class QuietShiftDetector:

    def __init__(
        self,
        baseline_window=20,
        recent_window=8,
        change_ratio=1.30,
        confirmations_required=3
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.change_ratio = change_ratio
        self.confirmations_required = confirmations_required

        self.error_history = []

        self.watch_count = 0
        self.change_confirmed = False

        self.confirmed_step = None
        self.confirmed_reason = None

    def update(self, step, error):

        error = float(error)

        self.error_history.append(error)

        # --------------------------------------------------
        # BUILD INITIAL BASELINE
        # --------------------------------------------------

        if len(self.error_history) < (
            self.baseline_window + self.recent_window
        ):

            return {
                "status": "NORMAL",
                "change_score": 0.0,
                "baseline_error": None,
                "recent_error": None,
                "trend": 0.0,
                "reason": "Building baseline",
                "change_confirmed": False,
                "confirmed_step": None
            }

        # --------------------------------------------------
        # CALCULATE BASELINE AND RECENT ERROR
        # --------------------------------------------------

        baseline_errors = self.error_history[
            -(
                self.baseline_window +
                self.recent_window
            ):
            -self.recent_window
        ]

        recent_errors = self.error_history[
            -self.recent_window:
        ]

        baseline_error = np.mean(
            baseline_errors
        )

        recent_error = np.mean(
            recent_errors
        )

        # Avoid division by zero
        baseline_error = max(
            baseline_error,
            1.0
        )

        # --------------------------------------------------
        # ERROR RATIO
        # --------------------------------------------------

        error_ratio = (
            recent_error /
            baseline_error
        )

        # --------------------------------------------------
        # ERROR TREND
        # --------------------------------------------------

        x = np.arange(
            len(recent_errors)
        )

        trend = np.polyfit(
            x,
            recent_errors,
            1
        )[0]

        # --------------------------------------------------
        # CHANGE SCORE
        # --------------------------------------------------

        ratio_score = max(
            0,
            error_ratio - 1
        )

        trend_score = max(
            0,
            trend / baseline_error
        )

        change_score = (
            0.7 * ratio_score +
            0.3 * trend_score
        )

        # --------------------------------------------------
        # DETECT SUSTAINED CHANGE
        # --------------------------------------------------

        evidence = (
            error_ratio >= self.change_ratio
            and trend > 0
        )

        if evidence:

            self.watch_count += 1

        else:

            self.watch_count = 0

        # --------------------------------------------------
        # CONFIRM CHANGE
        # --------------------------------------------------

        if (
            self.watch_count >=
            self.confirmations_required
            and not self.change_confirmed
        ):

            self.change_confirmed = True

            self.confirmed_step = step

            self.confirmed_reason = (
                "Recent rolling error remained "
                "consistently above the historical "
                "baseline with an increasing trend."
            )

            return {
                "status": "CONFIRMED CHANGE",
                "change_score": change_score,
                "baseline_error": baseline_error,
                "recent_error": recent_error,
                "trend": trend,
                "reason": self.confirmed_reason,
                "change_confirmed": True,
                "confirmed_step": step
            }

        # --------------------------------------------------
        # WATCH STATE
        # --------------------------------------------------

        if evidence:

            return {
                "status": "WATCH",
                "change_score": change_score,
                "baseline_error": baseline_error,
                "recent_error": recent_error,
                "trend": trend,
                "reason": (
                    "Recent error is increasing "
                    "above the historical baseline. "
                    "Waiting for sustained evidence."
                ),
                "change_confirmed": False,
                "confirmed_step": None
            }

        # --------------------------------------------------
        # NORMAL STATE
        # --------------------------------------------------

        return {
            "status": "NORMAL",
            "change_score": change_score,
            "baseline_error": baseline_error,
            "recent_error": recent_error,
            "trend": trend,
            "reason": (
                "Recent forecasting behaviour "
                "is consistent with the baseline."
            ),
            "change_confirmed": False,
            "confirmed_step": None
        }