import numpy as np


def calculate_error(actual, predicted):
    return abs(actual - predicted)


def calculate_change_score(errors, window=10):

    if len(errors) < window * 2:
        return 0.0

    previous_errors = errors[-2 * window:-window]
    recent_errors = errors[-window:]

    previous_mean = np.mean(previous_errors)
    recent_mean = np.mean(recent_errors)

    if previous_mean == 0:
        return 0.0

    return recent_mean / previous_mean


def change_detected(change_score, threshold=1.5):
    return change_score >= threshold