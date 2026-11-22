
"""
Live Flow Telemetry Logger.

Records every completed network flow, including:
- Normal flows
- Anomalous flows
- Detection errors

This provides persistent monitoring data for the dashboard.
"""

import csv
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TELEMETRY_FILE = (
    PROJECT_ROOT / "logs" / "live_flow_telemetry.csv"
)


# --------------------------------------------------
# CSV SCHEMA
# --------------------------------------------------

TELEMETRY_COLUMNS = [
    "event_time",
    "source_ip",
    "source_port",
    "destination_ip",
    "destination_port",
    "protocol",
    "duration_seconds",
    "packet_count",
    "total_bytes",
    "packet_rate",
    "byte_rate",
    "prediction",
    "is_anomaly",
    "anomaly_score",
    "decision_score",
    "threat_score",
    "threat_level",
    "detection_status",
]


# --------------------------------------------------
# TELEMETRY LOGGER
# --------------------------------------------------

class TelemetryLogger:

    def __init__(self, telemetry_file=TELEMETRY_FILE):

        self.telemetry_file = Path(telemetry_file)

        # Create the logs directory if it does not exist.
        self.telemetry_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def record_flow(
        self,
        flow,
        detection_result=None,
        threat_result=None,
        detection_status="SUCCESS"
    ):
        """
        Append one completed flow to the telemetry CSV.

        Normal and anomalous flows are both recorded.
        """

        detection_result = detection_result or {}
        threat_result = threat_result or {}

        row = {
            "event_time": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),

            "source_ip": flow.get("source_ip"),
            "source_port": flow.get("source_port"),

            "destination_ip": flow.get("destination_ip"),
            "destination_port": flow.get("destination_port"),

            "protocol": flow.get("protocol"),

            "duration_seconds": flow.get("duration_seconds"),
            "packet_count": flow.get("packet_count"),
            "total_bytes": flow.get("total_bytes"),

            "packet_rate": flow.get("packet_rate"),
            "byte_rate": flow.get("byte_rate"),

            "prediction": detection_result.get(
                "prediction", "ERROR"
            ),

            "is_anomaly": detection_result.get(
                "is_anomaly", ""
            ),

            "anomaly_score": detection_result.get(
                "anomaly_score", ""
            ),

            "decision_score": detection_result.get(
                "decision_score", ""
            ),

            "threat_score": threat_result.get(
                "threat_score", ""
            ),

            "threat_level": threat_result.get(
                "threat_level", "UNSCORED"
            ),

            "detection_status": detection_status,
        }

        # Check whether the CSV already contains a header.
        file_exists = (
            self.telemetry_file.exists()
            and self.telemetry_file.stat().st_size > 0
        )

        # Append the new flow without deleting previous records.
        with self.telemetry_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=TELEMETRY_COLUMNS
            )

            # Write the header only when creating a new file.
            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        return True
