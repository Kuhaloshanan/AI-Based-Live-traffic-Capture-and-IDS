
import csv
import tempfile
import unittest
from pathlib import Path

from src.live_rf_window_processor import RF_FEATURE_COLUMNS
from src.live_rf_telemetry_logger import LiveRFTelemetryLogger


class TestLiveRFTelemetryLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        log_path = Path(self.temp_dir.name) / "logs" / "test.csv"

        self.logger = LiveRFTelemetryLogger(log_path)

        # Create a sample 10-packet feature record.
        features = {
            name: float(index)
            for index, name in enumerate(RF_FEATURE_COLUMNS)
        }

        features["Number"] = 10.0

        self.window = {
            "window_size": 10,
            "features": features
        }

        self.prediction = {
            "prediction": "BENIGN",
            "window_size": 10,
            "is_benign": True,
            "model_score": 0.956,
            "top_predictions": [
                {"category": "BENIGN", "model_score": 0.956},
                {"category": "MITM-ARPSPOOFING", "model_score": 0.0248},
                {"category": "DNS_SPOOFING", "model_score": 0.0102},
            ]
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prediction_is_saved(self):
        self.logger.record_prediction(
            self.window,
            self.prediction
        )

        with self.logger.log_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "BENIGN")
        self.assertEqual(rows[0]["window_size"], "10")
        self.assertEqual(rows[0]["is_benign"], "True")
        self.assertAlmostEqual(
            float(rows[0]["model_score"]),
            0.956
        )

    def test_header_is_not_duplicated(self):
        self.logger.record_prediction(
            self.window,
            self.prediction
        )

        self.logger.record_prediction(
            self.window,
            self.prediction
        )

        with self.logger.log_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        # One header plus two prediction records.
        self.assertEqual(len(lines), 3)

    def test_invalid_feature_schema_is_rejected(self):
        invalid_window = {
            "window_size": 10,
            "features": {"wrong_feature": 1}
        }

        with self.assertRaises(ValueError):
            self.logger.record_prediction(
                invalid_window,
                self.prediction
            )


if __name__ == "__main__":
    unittest.main()
