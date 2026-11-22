
import csv
import tempfile
import unittest

from pathlib import Path

from src.telemetry_logger import TelemetryLogger


class TestTelemetryLogger(unittest.TestCase):

    def setUp(self):

        self.temp_directory = tempfile.TemporaryDirectory()

        self.test_file = (
            Path(self.temp_directory.name)
            / "logs"
            / "test_telemetry.csv"
        )

        self.logger = TelemetryLogger(self.test_file)

        self.sample_flow = {
            "source_ip": "192.168.1.10",
            "source_port": 5000,
            "destination_ip": "192.168.1.20",
            "destination_port": 443,
            "protocol": "TCP",
            "duration_seconds": 2.5,
            "packet_count": 10,
            "total_bytes": 5000,
            "packet_rate": 4.0,
            "byte_rate": 2000.0,
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def read_records(self):

        with self.test_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            return list(csv.DictReader(file))

    def test_normal_flow_is_recorded(self):

        self.logger.record_flow(
            self.sample_flow,
            {
                "prediction": "NORMAL",
                "is_anomaly": False,
                "anomaly_score": -0.15,
                "decision_score": 0.15,
            },
            {
                "threat_score": 0,
                "threat_level": "LOW",
            }
        )

        records = self.read_records()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prediction"], "NORMAL")

    def test_anomalous_flow_is_recorded(self):

        self.logger.record_flow(
            self.sample_flow,
            {
                "prediction": "ANOMALY",
                "is_anomaly": True,
                "anomaly_score": 0.12,
                "decision_score": -0.12,
            },
            {
                "threat_score": 80,
                "threat_level": "CRITICAL",
            }
        )

        records = self.read_records()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prediction"], "ANOMALY")
        self.assertEqual(records[0]["threat_level"], "CRITICAL")

    def test_detection_error_is_recorded(self):

        self.logger.record_flow(
            self.sample_flow,
            detection_status="ERROR"
        )

        records = self.read_records()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prediction"], "ERROR")
        self.assertEqual(records[0]["detection_status"], "ERROR")

    def test_multiple_flows_are_appended(self):

        for _ in range(3):

            self.logger.record_flow(
                self.sample_flow,
                {
                    "prediction": "NORMAL",
                    "is_anomaly": False,
                }
            )

        records = self.read_records()

        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
