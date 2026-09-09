import os

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import (
    load_data,
    prepare_data
)

from src.model import ForecastingModel


DATA_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


def main():

    print("=" * 60)
    print("ADAPTIVE HOSPITAL DEMAND FORECASTING")
    print("VERSION 1 - MODEL TRAINING")
    print("=" * 60)

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------

    df = load_data(DATA_PATH)

    print("\nDataset shape:", df.shape)

    print("\nDate range:")

    if "timestamp" in df.columns:
        print(
            df["timestamp"].min(),
            "to",
            df["timestamp"].max()
        )

    # -----------------------------------------
    # Prepare data
    # -----------------------------------------

    X, y, feature_columns = prepare_data(df)

    print("\nFeatures:")
    print(feature_columns)

    print("\nTarget:")
    print("target")

    # -----------------------------------------
    # Chronological split
    # -----------------------------------------

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_test))

    # -----------------------------------------
    # Train model
    # -----------------------------------------

    model = ForecastingModel()

    print("\nTraining model...")

    model.train(
        X_train,
        y_train
    )

    # -----------------------------------------
    # Predictions
    # -----------------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # MAPE
    non_zero = y_test != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_test[non_zero]
                    - predictions[non_zero]
                )
                /
                y_test[non_zero]
            )
        ) * 100

    else:

        mape = 0

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("MODEL VALIDATION RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    # -----------------------------------------
    # Save model
    # -----------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    print("\nV1 training completed.")


if __name__ == "__main__":
    main()