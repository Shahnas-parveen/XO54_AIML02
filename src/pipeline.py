import pandas as pd

from src.forecaster import Forecaster
from src.monitor import (
    calculate_error,
    calculate_change_score,
    change_detected
)
from src.adapter import AdaptiveController

from config import (
    ERROR_WINDOW,
    CHANGE_THRESHOLD,
    MIN_ADAPTATION_POINTS,
    RANDOM_STATE
)


def run_forecasting(df, feature_columns, target_column="target"):

    X = df[feature_columns]
    y = df[target_column]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    forecaster = Forecaster(RANDOM_STATE)

    forecaster.train(X_train, y_train)

    adapter = AdaptiveController()

    predictions = []
    actuals = []
    errors = []
    change_scores = []
    statuses = []

    for i in range(len(X_test)):

        current_X = X_test.iloc[[i]]

        prediction = forecaster.predict(current_X)

        actual = float(y_test.iloc[i])

        error = calculate_error(actual, prediction)

        predictions.append(prediction)
        actuals.append(actual)
        errors.append(error)

        score = calculate_change_score(
            errors,
            ERROR_WINDOW
        )

        change_scores.append(score)

        if change_detected(
            score,
            CHANGE_THRESHOLD
        ):

            recent_start = max(
                0,
                len(X_train) - MIN_ADAPTATION_POINTS
            )

            X_recent = X_train.iloc[recent_start:]
            y_recent = y_train.iloc[recent_start:]

            adapter.adapt(
                forecaster,
                X_recent,
                y_recent
            )

            statuses.append("Adapted")

        else:
            statuses.append("Monitoring")

    results = pd.DataFrame({
        "timestamp": df.iloc[split_index:]["timestamp"].values,
        "forecast": predictions,
        "actual": actuals,
        "error": errors,
        "change_score": change_scores,
        "adaptation_status": statuses
    })

    return results