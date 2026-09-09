import numpy as np


def calculate_error(actual, predicted):

    error = actual - predicted

    absolute_error = abs(error)

    if actual != 0:
        percentage_error = (
            abs(error) / abs(actual)
        ) * 100
    else:
        percentage_error = 0

    return {
        "error": error,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error
    }


def get_basic_status(error_history):

    if len(error_history) < 5:
        return "NORMAL"

    recent_errors = np.array(error_history[-5:])

    mean_error = np.mean(recent_errors)

    if mean_error < 10:
        return "NORMAL"

    return "HIGH ERROR"