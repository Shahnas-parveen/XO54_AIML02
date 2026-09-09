import pandas as pd


TARGET_COLUMN = "target"


def load_data(path):
    df = pd.read_csv(path)

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort chronologically
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_data(df):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    # Input features
    X = df[feature_columns].copy()

    # Target
    y = df[TARGET_COLUMN].copy()

    # Convert features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    X = X.ffill().bfill()

    # Remaining missing values
    X = X.fillna(X.median())

    return X, y, feature_columns