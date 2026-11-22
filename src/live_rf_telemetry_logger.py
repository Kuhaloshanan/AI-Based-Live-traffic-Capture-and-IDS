
"""
Persistent telemetry logger for live Random Forest predictions.

Stores each completed packet-window prediction in a CSV file.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from src.live_rf_window_processor import RF_FEATURE_COLUMNS


# Store telemetry in the project's logs directory.
DEFAULT_LOG_FILE = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "live_rf_predictions.csv"
)

BASE_COLUMNS = [
    "event_time",
    "window_size",
    "prediction",
    "is_benign",
    "model_score",
    "top_predictions",
]

# Include the exact 39 features supplied to the model.
CSV_COLUMNS = BASE_COLUMNS + RF_FEATURE_COLUMNS


class LiveRFTelemetryLogger:

    def __init__(self, log_file=DEFAULT_LOG_FILE):
        self.log_file = Path(log_file)

        # Create the logs directory if it does not exist.
        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def record_prediction(self, window_result, prediction_result):
        """
        Validate and append one Random Forest prediction.

        window_result:
            Completed packet window containing window_size and features.

        prediction_result:
            Prediction returned by LiveRandomForestEngine.
        """

        features = window_result["features"]

        # Check that the model input has the exact expected schema.
        if list(features.keys()) != RF_FEATURE_COLUMNS:
            raise ValueError(
                "Random Forest telemetry feature schema mismatch."
            )

        window_size = int(window_result["window_size"])

        if window_size not in (10, 100):
            raise ValueError("Unsupported packet window size.")

        if int(features["Number"]) != window_size:
            raise ValueError(
                "Feature Number does not match the packet window size."
            )

        # Validate every model input before writing the record.
        clean_features = {}

        for name in RF_FEATURE_COLUMNS:
            try:
                value = float(features[name])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid numeric feature: {name}"
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite feature value: {name}"
                )

            clean_features[name] = value

        prediction = str(prediction_result["prediction"])
        is_benign = bool(prediction_result["is_benign"])
        model_score = float(prediction_result["model_score"])

        if not math.isfinite(model_score):
            raise ValueError("Model score must be finite.")

        if int(prediction_result["window_size"]) != window_size:
            raise ValueError(
                "Prediction window size does not match input window."
            )

        top_predictions = prediction_result["top_predictions"]

        # Serialize the top predictions into one CSV cell.
        top_predictions_json = json.dumps(
            top_predictions,
            ensure_ascii=False
        )

        record = {
            "event_time": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "window_size": window_size,
            "prediction": prediction,
            "is_benign": is_benign,
            "model_score": model_score,
            "top_predictions": top_predictions_json,
            **clean_features,
        }

        # Write the header only when creating a new or empty file.
        write_header = (
            not self.log_file.exists()
            or self.log_file.stat().st_size == 0
        )

        with self.log_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_COLUMNS
            )

            if write_header:
                writer.writeheader()

            writer.writerow(record)
