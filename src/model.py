import joblib

from sklearn.ensemble import RandomForestRegressor


class ForecastingModel:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        joblib.dump(
            self.model,
            path
        )

    def load(self, path):

        self.model = joblib.load(path)