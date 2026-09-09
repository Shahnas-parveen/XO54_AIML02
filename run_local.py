import pandas as pd

from src.preprocessing import load_data, prepare_data
from src.model import ForecastingModel
from src.pipeline import ForecastPipeline


DATA_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


def main():

    print("=" * 60)
    print("LOCAL FORECASTING SIMULATION")
    print("=" * 60)

    # Load dataset
    df = load_data(DATA_PATH)

    X, y, feature_columns = prepare_data(df)

    # Load trained model
    model = ForecastingModel()
    model.load(MODEL_PATH)

    pipeline = ForecastPipeline(model)

    print("\nStarting sequential simulation...\n")

    # Use last 20% as simulation/evaluation period
    start_index = int(len(df) * 0.8)

    for i in range(start_index, len(df)):

        features = X.iloc[[i]]

        # Model predicts BEFORE seeing actual
        prediction = pipeline.predict(features)

        # Actual becomes available afterwards
        actual = float(y.iloc[i])

        timestamp = (
            df.iloc[i]["timestamp"]
            if "timestamp" in df.columns
            else i
        )

        record = pipeline.update(
            step=i,
            timestamp=timestamp,
            prediction=prediction,
            actual=actual
        )

        print(
            f"Step {i} | "
            f"Prediction: {prediction:.2f} | "
            f"Actual: {actual:.2f} | "
            f"Error: {record['absolute_error']:.2f}"
        )

    print("\nSimulation completed.")

    print("\nLog saved to:")
    print("logs/evaluation_log.csv")


if __name__ == "__main__":
    main()