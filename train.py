import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.features import (
    add_features,
    get_feature_columns
)
from src.model import ForecastingModel


DATA_PATH = "data/training_set.csv"
MODEL_PATH = "models/forecasting_model.pkl"


print("=" * 60)
print("ADAPTIVE HOSPITAL DEMAND FORECASTING")
print("IMPROVED MODEL - LAG + ROLLING FEATURES")
print("=" * 60)


df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("\nDate range:")
print(df["timestamp"].min(), "to", df["timestamp"].max())


df = add_features(df)

feature_columns = get_feature_columns()

df = df.dropna().reset_index(drop=True)

X = df[feature_columns]
y = df["target"]


split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("\nFeature count:", len(feature_columns))
print("Training samples:", len(X_train))
print("Validation samples:", len(X_test))


print("\nTraining improved model...")

model = ForecastingModel()

model.train(
    X_train,
    y_train
)

predictions = model.predict(X_test)


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

nonzero = y_test != 0

mape = (
    np.mean(
        np.abs(
            (
                y_test[nonzero]
                - predictions[nonzero]
            )
            / y_test[nonzero]
        )
    ) * 100
)


print("\n" + "=" * 60)
print("IMPROVED MODEL VALIDATION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")


# Final training on ALL available historical data
print("\nTraining final model on all historical data...")

final_model = ForecastingModel()

final_model.train(
    X,
    y
)

os.makedirs(
    "models",
    exist_ok=True
)

final_model.save(
    MODEL_PATH
)

print("\nFinal model saved successfully:")
print(MODEL_PATH)

print("\nTraining completed.")