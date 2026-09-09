import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    return df


def prepare_data(df):
    """
    Prepare training data.

    Expected structure:
    feature_1, feature_2, ..., feature_n, target
    """

    # Target column
    target_column = "target"

    if target_column not in df.columns:
        raise ValueError(
            f"'target' column not found. Available columns: {list(df.columns)}"
        )

    # Automatically identify feature columns
    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    if not feature_columns:
        raise ValueError("No feature_* columns found.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    X = X.ffill().bfill()

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill any remaining missing values
    X = X.fillna(X.median())

    return X, y, feature_columns