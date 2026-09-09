import pandas as pd


BASE_FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
]

LAGS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 14]


def add_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Original features
    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calendar features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Previous target values
    for lag in LAGS:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # Rolling target statistics using only previous observations
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = (
            df["target"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    return df


def get_feature_columns():
    return (
        BASE_FEATURES
        + [
            "day_of_week",
            "month",
            "day_of_year",
        ]
        + [f"lag_{lag}" for lag in LAGS]
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    )


def build_live_features(timestamp, features, target_history):
    row = {
        **features
    }

    timestamp = pd.to_datetime(timestamp)

    row["day_of_week"] = timestamp.dayofweek
    row["month"] = timestamp.month
    row["day_of_year"] = timestamp.dayofyear

    history = list(target_history)

    for lag in LAGS:
        if len(history) >= lag:
            row[f"lag_{lag}"] = history[-lag]
        else:
            row[f"lag_{lag}"] = history[0]

    for window in ROLLING_WINDOWS:
        values = history[-window:]

        if len(values) == 0:
            row[f"roll_mean_{window}"] = 0.0
        else:
            row[f"roll_mean_{window}"] = sum(values) / len(values)

    return pd.DataFrame(
        [[row[col] for col in get_feature_columns()]],
        columns=get_feature_columns()
    )