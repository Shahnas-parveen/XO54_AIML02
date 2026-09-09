from src.preprocessing import (
    load_data,
    prepare_data
)

from src.model import ForecastingModel

from src.pipeline import ForecastPipeline


DATA_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


def main():

    print("=" * 60)
    print("LOCAL SEQUENTIAL FORECASTING")
    print("=" * 60)

    # Load data
    df = load_data(DATA_PATH)

    X, y, feature_columns = prepare_data(df)

    # Load trained model
    model = ForecastingModel()

    model.load(MODEL_PATH)

    # Create pipeline
    pipeline = ForecastPipeline(model)

    # Last 20% used as simulation period
    start_index = int(
        len(df) * 0.8
    )

    print(
        f"\nSimulation starts from row {start_index}"
    )

    print(
        "Simulating predictions sequentially...\n"
    )

    for i in range(
        start_index,
        len(df)
    ):

        # Features available BEFORE actual target
        features = X.iloc[[i]]

        # Make prediction
        prediction = pipeline.predict(
            features
        )

        # Actual becomes available
        actual = float(
            y.iloc[i]
        )

        # Timestamp
        timestamp = df.iloc[i]["timestamp"]

        # Update monitoring
        record = pipeline.update(
            step=i,
            timestamp=timestamp,
            prediction=prediction,
            actual=actual
        )

        print(
            f"{timestamp.date()} | "
            f"Prediction: {prediction:.2f} | "
            f"Actual: {actual:.2f} | "
            f"Error: {record['absolute_error']:.2f} | "
            f"Status: {record['status']}"
        )

    print(
        "\nSimulation completed successfully."
    )

    print(
        "\nLog saved to:"
    )

    print(
        "logs/evaluation_log.csv"
    )


if __name__ == "__main__":
    main()