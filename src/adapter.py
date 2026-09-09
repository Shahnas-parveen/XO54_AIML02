class AdaptiveController:

    def __init__(self):
        self.adaptation_count = 0

    def adapt(self, forecaster, X_recent, y_recent):

        forecaster.train(X_recent, y_recent)

        self.adaptation_count += 1

        return forecaster