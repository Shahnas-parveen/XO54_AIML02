import requests


BASE_URL = "https://ps2-eval-server.onrender.com"


class EvaluationAPI:

    def __init__(self, team_name, stream, session_id=None):

        self.team_name = team_name
        self.stream = stream
        self.session_id = session_id

    def start_session(self):

        if self.session_id is not None:
            raise RuntimeError(
                "Session already exists. "
                "Do not start another session."
            )

        response = requests.post(
            f"{BASE_URL}/session/start",
            json={
                "team_name": self.team_name,
                "stream": self.stream
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        self.session_id = data["session_id"]

        return data

    def get_next(self):

        if self.session_id is None:
            raise RuntimeError(
                "Session has not been started."
            )

        response = requests.get(
            f"{BASE_URL}/session/"
            f"{self.session_id}/next",
            timeout=30
        )

        # Show the server message if a prediction is already pending
        if response.status_code == 409:

            print("\n" + "=" * 60)
            print("API 409 RESPONSE")
            print("=" * 60)
            print(response.text)
            print("=" * 60)

            raise RuntimeError(
                "The session already has a pending prediction."
            )

        response.raise_for_status()

        return response.json()

    def submit_prediction(self, prediction):

        if self.session_id is None:
            raise RuntimeError(
                "Session has not been started."
            )

        response = requests.post(
            f"{BASE_URL}/session/"
            f"{self.session_id}/predict",
            json={
                "predicted_target": float(prediction)
            },
            timeout=30
        )

        response.raise_for_status()

        return response.json()