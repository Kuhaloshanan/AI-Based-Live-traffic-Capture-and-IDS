
import csv
import json
import tempfile
import unittest

from pathlib import Path

from src.unified_telemetry_logger import UnifiedTelemetryLogger


class TestUnifiedTelemetryLogger(unittest.TestCase):

    def setUp(self):

        self.temp_directory = tempfile.TemporaryDirectory()

        self.test_file = (
            Path(self.temp_directory.name)
            / "unified_decisions.csv"
        )

        self.logger = UnifiedTelemetryLogger(
            self.test_file
        )

        self.flow_key = (
            "TCP",
            ("192.168.1.10", 50000),
            ("1.1.1.1", 443)
        )

        self.flow = {
            "flow_key": self.flow_key,
            "source_ip": "192.168.1.10",
            "source_port": 50000,
            "destination_ip": "1.1.1.1",
            "destination_port": 443,
            "protocol": "TCP",
            "duration_seconds": 5.0,
            "packet_count": 20,
            "total_bytes": 12000,
            "packet_rate": 4.0,
            "byte_rate": 2400.0,
        }

        self.unified_result = {
            "flow_key": self.flow_key,
            "final_prediction": "DDoS",
            "threat_score": 70,
            "threat_level": "HIGH",
            "evidence_status": "BOTH_MODELS_FLAGGED",
            "is_attack_flagged": True,
            "rf_prediction": "DDoS",
            "rf_window_count": 2,
            "rf_benign_window_count": 0,
            "rf_flagged_window_count": 2,
            "rf_category_counts": {"DDoS": 2},
            "rf_window_predictions": [
                {
                    "window_size": 10,
                    "prediction": "DDoS",
                    "is_benign": False,
                    "model_score": 0.91,
                }
            ],
            "if_prediction": "ANOMALY",
            "if_is_anomaly": True,
            "if_threat_score": 35,
            "score_reasons": [
                "Random Forest flagged the flow.",
                "Isolation Forest detected an anomaly.",
            ],
            "classification_note": "Test decision.",
        }

        self.isolation_result = {
            "prediction": "ANOMALY",
            "is_anomaly": True,
            "anomaly_score": 0.12,
            "decision_score": -0.12,
            "raw_score": -0.55,
        }

        self.threat_result = {
            "threat_score": 35,
            "threat_level": "MEDIUM",
            "score_reasons": ["Isolation Forest anomaly."],
            "baseline_percentiles": {
                "duration_seconds": 99.2,
                "packet_rate": 97.5,
            },
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def read_records(self):

        with self.test_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            return list(csv.DictReader(csv_file))

    # -------------------------------------------------
    # TEST 1: ONE UNIFIED DECISION IS SAVED
    # -------------------------------------------------

    def test_unified_decision_is_saved(self):

        result = self.logger.record_decision(
            self.flow,
            self.unified_result,
            self.isolation_result,
            self.threat_result
        )

        self.assertTrue(result)
        self.assertTrue(self.test_file.exists())

        records = self.read_records()

        self.assertEqual(len(records), 1)

        self.assertEqual(
            records[0]["final_prediction"],
            "DDoS"
        )

        self.assertEqual(
            records[0]["threat_level"],
            "HIGH"
        )

    # -------------------------------------------------
    # TEST 2: BOTH MODEL EVIDENCE SOURCES ARE SAVED
    # -------------------------------------------------

    def test_model_evidence_is_preserved(self):

        self.logger.record_decision(
            self.flow,
            self.unified_result,
            self.isolation_result,
            self.threat_result
        )

        records = self.read_records()
        record = records[0]

        self.assertEqual(
            record["rf_prediction"],
            "DDoS"
        )

        self.assertEqual(
            record["if_prediction"],
            "ANOMALY"
        )

        self.assertEqual(
            record["rf_window_count"],
            "2"
        )

        self.assertEqual(
            record["if_is_anomaly"],
            "True"
        )

        # Confirm structured JSON evidence can be recovered.
        categories = json.loads(
            record["rf_category_counts"]
        )

        self.assertEqual(categories["DDoS"], 2)

    # -------------------------------------------------
    # TEST 3: MULTIPLE DECISIONS ARE APPENDED
    # -------------------------------------------------

    def test_multiple_decisions_are_appended(self):

        self.logger.record_decision(
            self.flow,
            self.unified_result,
            self.isolation_result,
            self.threat_result
        )

        self.logger.record_decision(
            self.flow,
            self.unified_result,
            self.isolation_result,
            self.threat_result
        )

        records = self.read_records()

        self.assertEqual(len(records), 2)

    # -------------------------------------------------
    # TEST 4: FLOW KEY MISMATCH IS REJECTED
    # -------------------------------------------------

    def test_flow_key_mismatch_is_rejected(self):

        invalid_result = self.unified_result.copy()

        invalid_result["flow_key"] = (
            "UDP",
            ("10.0.0.1", 1000),
            ("10.0.0.2", 2000)
        )

        with self.assertRaises(ValueError):

            self.logger.record_decision(
                self.flow,
                invalid_result,
                self.isolation_result,
                self.threat_result
            )

    # -------------------------------------------------
    # TEST 5: MISSING FLOW KEY IS REJECTED
    # -------------------------------------------------

    def test_missing_flow_key_is_rejected(self):

        invalid_flow = self.flow.copy()
        del invalid_flow["flow_key"]

        with self.assertRaises(ValueError):

            self.logger.record_decision(
                invalid_flow,
                self.unified_result
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
