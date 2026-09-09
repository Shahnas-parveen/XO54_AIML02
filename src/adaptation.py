class AdaptiveTrainer:

    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

    def adapt(self, observed_data, recent_window=50):

        if len(observed_data) < 10:
            return False

        recent_data = observed_data.tail(
            recent_window
        ).copy()

        X = recent_data[
            self.feature_columns
        ]

        y = recent_data["target"]

        self.model.train(X, y)

        return True