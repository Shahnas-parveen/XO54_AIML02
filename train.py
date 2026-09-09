import os

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from src.preprocessing import load_data, prepare_data
from src.model import ForecastingModel


DATA_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


def main():

    print("=" * 60)
    print("ADAPTIVE HOSPITAL DEMAND FORECASTING")
    print("MODEL TRAINING")
    print("=" * 60)

    # Load data
    df = load_data(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(list(df.columns))

    # Prepare data
    X, y, feature_columns = prepare_data(df)

    print("\nFeatures:")
    print(feature_columns)

    print("\nTarget:")
    print("target")

    # Time-based split
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_test))

    # Train
    model = ForecastingModel()

    print("\nTraining model...")
    model.train(X_train, y_train)

    # Validate
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # MAPE
    non_zero = y_test != 0

    if non_zero.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (y_test[non_zero] - predictions[non_zero])
                    / y_test[non_zero]
                )
            )
            * 100
        )
    else:
        mape = 0

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)

    model.save(MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()