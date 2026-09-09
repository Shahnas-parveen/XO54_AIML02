import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class PatientForecaster:

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    def create_features(self, df):

        data = df.copy()

        for i in range(1, 8):
            data[f"lag_{i}"] = data["target"].shift(i)

        data = data.dropna()

        return data

    def train(self, df):

        data = self.create_features(df)

        feature_columns = [
            f"lag_{i}" for i in range(1, 8)
        ]

        X = data[feature_columns]
        y = data["target"]

        self.model.fit(X, y)

    def predict_next(self, df):

        latest_values = df["target"].tail(7).values

        features = {}

        for i in range(1, 8):
            features[f"lag_{i}"] = latest_values[-i]

        X = pd.DataFrame([features])

        prediction = self.model.predict(X)[0]

        return float(prediction)