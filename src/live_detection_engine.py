
"""
Live AI Detection Engine.

Loads the trained Isolation Forest model and uses it
to identify anomalous network flows.

This component does not capture packets itself.
It receives completed flow dictionaries from FlowProcessor.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Locate the trained model relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "live_isolation_forest.pkl"
)


class LiveDetectionEngine:
    """
    Predict whether a completed network flow is anomalous.

    The model expects eight numerical flow features.
    Protocol and IP addresses are not model inputs.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH):

        self.model_path = Path(model_path)

        # Ensure the trained model exists.
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Load the saved model artifact.
        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]

        # Define the expected input features.
        expected_features = [
            "duration_seconds",
            "forward_packets",
            "reverse_packets",
            "forward_bytes",
            "reverse_bytes",
            "min_packet_size",
            "max_packet_size",
            "standard_deviation",
        ]

        # Prevent accidental use of an incompatible model.
        if self.feature_columns != expected_features:
            raise ValueError(
                "The saved model feature schema does not match "
                "the expected live flow schema."
            )

        print("Live AI Detection Engine initialized.")
        print(f"Loaded model: {self.model_path.name}")
        print(f"Expected features: {len(self.feature_columns)}")

    def extract_features(self, flow):
        """
        Extract and validate the eight model input features.

        The flow may contain additional metadata such as
        IP addresses, ports and protocol. These are ignored
        by the current Isolation Forest model.
        """

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in flow
        ]

        if missing_features:
            raise ValueError(
                f"Missing flow features: {missing_features}"
            )

        # Build a DataFrame with the exact model feature order.
        feature_values = {
            feature: flow[feature]
            for feature in self.feature_columns
        }

        features = pd.DataFrame(
            [feature_values],
            columns=self.feature_columns
        )

        # Convert inputs to numerical values.
        try:
            features = features.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Flow features must be numerical."
            ) from error

        # Reject NaN and infinite values.
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError(
                "Flow contains missing or infinite feature values."
            )

        # Network traffic statistics should not be negative.
        if (features < 0).any().any():
            raise ValueError(
                "Flow features cannot contain negative values."
            )

        return features

    def predict_flow(self, flow):
        """
        Predict whether a completed network flow is anomalous.

        Isolation Forest:
            +1 = Normal
            -1 = Anomaly

        The anomaly margin is the negative decision_function.
        Positive values indicate an anomaly.
        """

        # Prepare the flow for model inference.
        features = self.extract_features(flow)

        # Obtain the model prediction.
        prediction = int(
            self.model.predict(features)[0]
        )

        # Obtain the raw Isolation Forest score.
        raw_score = float(
            self.model.score_samples(features)[0]
        )

        # Obtain the score relative to the model threshold.
        decision_score = float(
            self.model.decision_function(features)[0]
        )

        is_anomaly = prediction == -1

        return {
            "prediction": (
                "ANOMALY" if is_anomaly else "NORMAL"
            ),
            "is_anomaly": is_anomaly,
            "anomaly_score": round(-decision_score, 6),
            "decision_score": round(decision_score, 6),
            "raw_score": round(raw_score, 6),
        }
