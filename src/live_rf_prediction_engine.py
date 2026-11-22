
"""Random Forest prediction engine for live IPv4 packet windows."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.live_rf_window_processor import RF_FEATURE_COLUMNS


# Path to the trained hybrid Random Forest model.
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "random_forest_live_hybrid.pkl"
)


class LiveRandomForestEngine:

    def __init__(self, model_path=MODEL_PATH):

        print("Loading live Random Forest model...")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Random Forest model not found: {model_path}"
            )

        # Load the trained model.
        self.model = joblib.load(model_path)

        # Verify the model's expected feature names and order.
        model_features = list(self.model.feature_names_in_)

        if model_features != RF_FEATURE_COLUMNS:
            raise ValueError(
                "Random Forest feature schema mismatch."
            )

        print("Live Random Forest loaded successfully.")
        print("Expected features:", len(model_features))
        print("Supported classes:", len(self.model.classes_))

    def predict_window(self, window_result):
        """
        Classify one completed IPv4 packet window.

        Expected input:
        {
            "window_size": 10 or 100,
            "features": {39 feature values}
        }
        """

        window_size = int(window_result["window_size"])
        features = window_result["features"]

        # Accept only the window sizes used during training.
        if window_size not in (10, 100):
            raise ValueError(
                f"Unsupported window size: {window_size}"
            )

        # Check for missing or unexpected features.
        missing = (
            set(RF_FEATURE_COLUMNS) - set(features.keys())
        )

        extra = (
            set(features.keys()) - set(RF_FEATURE_COLUMNS)
        )

        if missing or extra:
            raise ValueError(
                f"Feature mismatch. Missing={missing}, Extra={extra}"
            )

        # Arrange features in the exact training order.
        input_data = pd.DataFrame(
            [[features[name] for name in RF_FEATURE_COLUMNS]],
            columns=RF_FEATURE_COLUMNS
        )

        # Convert all input values to numeric.
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        values = input_data.to_numpy(dtype=float)

        # Reject missing or infinite values.
        if not np.isfinite(values).all():
            raise ValueError(
                "Non-finite feature values received."
            )

        # This model currently supports IPv4 windows only.
        if float(features["IPv"]) != 1:
            raise ValueError(
                "Random Forest currently supports IPv4 windows only."
            )

        # Verify that the packet count matches the window size.
        if int(round(float(features["Number"]))) != window_size:
            raise ValueError(
                "Number feature does not match window size."
            )

        # Predict the traffic category.
        prediction = str(
            self.model.predict(input_data)[0]
        )

        # Obtain the model's class scores.
        # These scores are not calibrated probabilities.
        scores = self.model.predict_proba(input_data)[0]

        # Find the three highest-scoring categories.
        top_indices = np.argsort(scores)[::-1][:3]

        top_predictions = [
            {
                "category": str(self.model.classes_[index]),
                "model_score": round(float(scores[index]), 4)
            }
            for index in top_indices
        ]

        # Return the prediction and supporting information.
        return {
            "prediction": prediction,
            "window_size": window_size,
            "is_benign": prediction == "BENIGN",
            "model_score": round(float(np.max(scores)), 4),
            "top_predictions": top_predictions
        }


if __name__ == "__main__":

    print("\n========== LIVE RANDOM FOREST ENGINE TEST ==========")

    engine = LiveRandomForestEngine()

    print("\nModel classes:")

    for category in engine.model.classes_:
        print("-", category)

    print("\nRandom Forest engine initialization successful.")
