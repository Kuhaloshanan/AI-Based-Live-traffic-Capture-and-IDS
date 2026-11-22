
"""Test the live Random Forest using a real captured window."""

import pandas as pd

from src.live_rf_prediction_engine import LiveRandomForestEngine
from src.live_rf_window_processor import RF_FEATURE_COLUMNS


CAPTURE_FILE = (
    "dataset/live_rf_validation/"
    "live_rf_windows_20261122_012041.csv"
)


def main():

    print("\n========== LIVE RF PREDICTION TEST ==========")

    # Load a real captured packet window.
    dataframe = pd.read_csv(CAPTURE_FILE)

    if dataframe.empty:
        raise ValueError("The capture file contains no windows.")

    sample = dataframe.iloc[0]

    window_result = {
        "window_size": int(sample["window_size"]),
        "features": {
            feature: float(sample[feature])
            for feature in RF_FEATURE_COLUMNS
        }
    }

    print("Window size:", window_result["window_size"])
    print("Feature count:", len(window_result["features"]))

    # Initialize the trained Random Forest.
    engine = LiveRandomForestEngine()

    # Classify the captured window.
    result = engine.predict_window(window_result)

    print("\n========== PREDICTION RESULT ==========")

    print("Predicted category:", result["prediction"])
    print("Is benign:", result["is_benign"])
    print("Model score:", result["model_score"])

    print("\nTop three predictions:")

    for item in result["top_predictions"]:
        print(
            f"{item['category']}: "
            f"{item['model_score']:.4f}"
        )

    print("\nReal captured window prediction successful.")


if __name__ == "__main__":
    main()
