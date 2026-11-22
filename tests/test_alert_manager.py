
"""
Unit tests for the Security Alert Manager.

Tests:
1. Normal traffic is ignored.
2. Anomalous traffic is recorded.
3. Multiple alerts are appended without overwriting.
"""

import csv
import tempfile
import unittest

from pathlib import Path

from src.alert_manager import AlertManager


class TestAlertManager(unittest.TestCase):

    def setUp(self):
        # Use a temporary directory so tests do not
        # modify the project's actual alert history.
        self.temp_directory = tempfile.TemporaryDirectory()

        self.alert_file = (
            Path(self.temp_directory.name)
            / "security_alerts.csv"
        )

        self.manager = AlertManager(
            alert_file=self.alert_file
        )

        # Sample completed network flow.
        self.flow = {
            "source_ip": "192.168.1.10",
            "source_port": 50000,
            "destination_ip": "8.8.8.8",
            "destination_port": 443,
            "protocol": "TCP",
            "duration_seconds": 12.5,
            "packet_count": 50,
            "total_bytes": 25000,
        }

        # Sample AI anomaly detection result.
        self.anomaly_result = {
            "prediction": "ANOMALY",
            "is_anomaly": True,
            "anomaly_score": 0.05,
            "decision_score": -0.05,
            "raw_score": -0.55,
        }

        # Sample normal detection result.
        self.normal_result = {
            "prediction": "NORMAL",
            "is_anomaly": False,
            "anomaly_score": -0.02,
            "decision_score": 0.02,
            "raw_score": -0.48,
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_normal_traffic_is_ignored(self):

        result = self.manager.record_detection(
            self.flow,
            self.normal_result
        )

        self.assertFalse(result)

        # No alert CSV should be created for normal traffic.
        self.assertFalse(self.alert_file.exists())

    def test_anomaly_is_recorded(self):

        result = self.manager.record_detection(
            self.flow,
            self.anomaly_result
        )

        self.assertTrue(result)
        self.assertTrue(self.alert_file.exists())

        # Read the saved alert.
        with self.alert_file.open(
            mode="r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.DictReader(csv_file)
            alerts = list(reader)

        self.assertEqual(len(alerts), 1)

        self.assertEqual(
            alerts[0]["source_ip"],
            "192.168.1.10"
        )

        self.assertEqual(
            alerts[0]["protocol"],
            "TCP"
        )

        self.assertEqual(
            alerts[0]["status"],
            "UNREVIEWED"
        )

    def test_multiple_alerts_are_appended(self):

        # Record the first anomaly.
        self.manager.record_detection(
            self.flow,
            self.anomaly_result
        )

        # Record another anomaly with a different source.
        second_flow = self.flow.copy()
        second_flow["source_ip"] = "192.168.1.20"

        self.manager.record_detection(
            second_flow,
            self.anomaly_result
        )

        # Both alerts should exist in the CSV.
        with self.alert_file.open(
            mode="r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.DictReader(csv_file)
            alerts = list(reader)

        self.assertEqual(len(alerts), 2)

        self.assertEqual(
            alerts[0]["source_ip"],
            "192.168.1.10"
        )

        self.assertEqual(
            alerts[1]["source_ip"],
            "192.168.1.20"
        )

    def test_rf_only_detection_creates_alert(self):

        unified_result = {
            "final_prediction": "DDoS",
            "evidence_status": "RANDOM_FOREST_ONLY",
            "is_attack_flagged": True,
            "rf_prediction": "DDoS",
            "rf_window_count": 2,
            "rf_flagged_window_count": 2,
            "threat_score": 40,
            "threat_level": "MEDIUM",
            "score_reasons": ["Random Forest flagged the flow."]
        }

        isolation_result = {
            "prediction": "NORMAL",
            "is_anomaly": False,
            "anomaly_score": -0.02,
            "decision_score": 0.02,
            "raw_score": -0.48
        }

        result = self.manager.record_unified_detection(
            self.flow,
            unified_result,
            isolation_result,
            {"baseline_percentiles": {}}
        )

        self.assertTrue(result)

        with self.alert_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            alerts = list(csv.DictReader(file))

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["final_prediction"], "DDoS")
        self.assertEqual(alerts[0]["rf_prediction"], "DDoS")
        self.assertEqual(
            alerts[0]["evidence_status"],
            "RANDOM_FOREST_ONLY"
        )

    def test_both_models_detection_creates_alert(self):

        unified_result = {
            "final_prediction": "DDoS",
            "evidence_status": "BOTH_MODELS_FLAGGED",
            "is_attack_flagged": True,
            "rf_prediction": "DDoS",
            "rf_window_count": 3,
            "rf_flagged_window_count": 3,
            "threat_score": 70,
            "threat_level": "HIGH",
            "score_reasons": ["Both models flagged the flow."]
        }

        isolation_result = {
            "prediction": "ANOMALY",
            "is_anomaly": True,
            "anomaly_score": 0.12,
            "decision_score": -0.12,
            "raw_score": -0.55
        }

        result = self.manager.record_unified_detection(
            self.flow,
            unified_result,
            isolation_result,
            {"baseline_percentiles": {}}
        )

        self.assertTrue(result)

        with self.alert_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            alerts = list(csv.DictReader(file))

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["rf_prediction"], "DDoS")
        self.assertEqual(alerts[0]["if_prediction"], "ANOMALY")
        self.assertEqual(
            alerts[0]["evidence_status"],
            "BOTH_MODELS_FLAGGED"
        )

    def test_normal_unified_decision_is_ignored(self):

        unified_result = {
            "final_prediction": "BENIGN",
            "evidence_status": "BOTH_MODELS_BENIGN",
            "is_attack_flagged": False,
            "rf_prediction": "BENIGN",
            "rf_window_count": 2,
            "rf_flagged_window_count": 0,
            "threat_score": 0,
            "threat_level": "LOW",
            "score_reasons": []
        }

        result = self.manager.record_unified_detection(
            self.flow,
            unified_result
        )

        self.assertFalse(result)
        self.assertFalse(self.alert_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
