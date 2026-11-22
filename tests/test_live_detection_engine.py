
"""
Test the Live AI Detection Engine.

Uses an actual collected network flow to verify
that the trained Isolation Forest works correctly.
"""

from pathlib import Path
import unittest
import math

import pandas as pd

from src.live_detection_engine import LiveDetectionEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)


class TestLiveDetectionEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the detection engine once for all tests.
        cls.engine = LiveDetectionEngine()

        # Read one real flow from our collected dataset.
        df = pd.read_csv(BASELINE_FILE)

        cls.sample_flow = df.iloc[0].to_dict()

    def test_prediction_returns_expected_fields(self):
        # Pass a real network flow to the AI engine.
        result = self.engine.predict_flow(
            self.sample_flow
        )

        expected_fields = [
            "prediction",
            "is_anomaly",
            "anomaly_score",
            "decision_score",
            "raw_score",
        ]

        # Verify that all expected outputs exist.
        for field in expected_fields:
            self.assertIn(field, result)

    def test_prediction_is_valid(self):
        result = self.engine.predict_flow(
            self.sample_flow
        )

        # Prediction must be one of these two outcomes.
        self.assertIn(
            result["prediction"],
            ["NORMAL", "ANOMALY"]
        )

        self.assertIsInstance(
            result["is_anomaly"],
            bool
        )

        # Check that the prediction agrees with the flag.
        self.assertEqual(
            result["is_anomaly"],
            result["prediction"] == "ANOMALY"
        )

        # Check that all returned scores are finite numbers.
        for score_name in [
            "anomaly_score",
            "decision_score",
            "raw_score",
        ]:
            self.assertTrue(
                math.isfinite(result[score_name])
            )

        # A negative decision score means anomaly.
        self.assertEqual(
            result["is_anomaly"],
            result["decision_score"] < 0
        )

    def test_missing_feature_is_rejected(self):
        # Create a copy with one required feature removed.
        incomplete_flow = self.sample_flow.copy()

        incomplete_flow.pop("forward_packets")

        # The engine must reject incomplete input.
        with self.assertRaises(ValueError):
            self.engine.predict_flow(incomplete_flow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
