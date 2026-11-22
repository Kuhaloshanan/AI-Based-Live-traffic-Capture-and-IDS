
"""
Unit tests for the Threat Scoring Engine.

Uses a synthetic baseline so tests do not depend
on the size or contents of the real traffic dataset.
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.threat_scoring import ThreatScorer


class TestThreatScorer(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.baseline_path = (
            Path(self.temp_dir.name) / "test_baseline.csv"
        )

        # Create predictable baseline values.
        baseline = pd.DataFrame({
            "duration_seconds": range(1, 101),
            "packet_rate": range(10, 110),
            "byte_rate": range(100, 200),
        })

        baseline.to_csv(self.baseline_path, index=False)

        self.scorer = ThreatScorer(self.baseline_path)

        self.detection_result = {
            "is_anomaly": True
        }

        self.normal_result = {
            "is_anomaly": False
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normal_flow_receives_zero_score(self):

        result = self.scorer.score_flow(
            {},
            self.normal_result
        )

        self.assertEqual(result["threat_score"], 0)
        self.assertEqual(result["threat_level"], "LOW")

    def test_extreme_traffic_receives_critical_score(self):

        flow = {
            "duration_seconds": 150,
            "packet_rate": 150,
            "byte_rate": 250,
        }

        result = self.scorer.score_flow(
            flow,
            self.detection_result
        )

        self.assertEqual(result["threat_score"], 100)
        self.assertEqual(result["threat_level"], "CRITICAL")

    def test_multiple_unusual_metrics_receive_bonus(self):

        flow = {
            "duration_seconds": 96,
            "packet_rate": 5,
            "byte_rate": 150,
        }

        result = self.scorer.score_flow(
            flow,
            self.detection_result
        )

        # Base 20 + duration 10 + low packet rate 10
        # + multiple-characteristic bonus 10 = 50.
        self.assertEqual(result["threat_score"], 50)
        self.assertEqual(result["threat_level"], "HIGH")

    def test_missing_feature_is_rejected(self):

        flow = {
            "duration_seconds": 50,
            "packet_rate": 20,
        }

        with self.assertRaises(ValueError):

            self.scorer.score_flow(
                flow,
                self.detection_result
            )

    def test_negative_feature_is_rejected(self):

        flow = {
            "duration_seconds": -10,
            "packet_rate": 20,
            "byte_rate": 150,
        }

        with self.assertRaises(ValueError):

            self.scorer.score_flow(
                flow,
                self.detection_result
            )

    def test_percentile_information_is_returned(self):

        flow = {
            "duration_seconds": 150,
            "packet_rate": 150,
            "byte_rate": 250,
        }

        result = self.scorer.score_flow(
            flow,
            self.detection_result
        )

        self.assertIn(
            "duration_seconds",
            result["baseline_percentiles"]
        )

        self.assertIn(
            "packet_rate",
            result["baseline_percentiles"]
        )

        self.assertIn(
            "byte_rate",
            result["baseline_percentiles"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
