
"""
Security Alert Manager.

Supports:
1. Existing Isolation Forest anomaly alerts.
2. New unified Random Forest + Isolation Forest alerts.
3. Persistent CSV alert history.
4. Automatic migration of older CSV schemas.
"""

import csv
import json

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ALERT_FILE = (
    PROJECT_ROOT / "logs" / "security_alerts.csv"
)


class AlertManager:

    def __init__(self, alert_file=DEFAULT_ALERT_FILE):

        self.alert_file = Path(alert_file)

        self.alert_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Existing columns are retained.
        # New columns support unified AI decisions.
        self.fieldnames = [
            "alert_time",
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
            "anomaly_score",
            "decision_score",
            "raw_score",
            "final_prediction",
            "evidence_status",
            "rf_prediction",
            "rf_window_count",
            "rf_flagged_window_count",
            "if_prediction",
            "if_is_anomaly",
            "is_attack_flagged",
            "threat_score",
            "threat_level",
            "score_reasons",
            "baseline_percentiles",
            "status",
        ]

        self._migrate_existing_file()

    # =====================================================
    # MIGRATE EXISTING ALERT CSV
    # =====================================================

    def _migrate_existing_file(self):

        if (
            not self.alert_file.exists()
            or self.alert_file.stat().st_size == 0
        ):
            return

        with self.alert_file.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            if reader.fieldnames == self.fieldnames:
                return

            existing_rows = list(reader)

        # Write the upgraded schema to a temporary file.
        temporary_file = self.alert_file.with_name(
            self.alert_file.name + ".tmp"
        )

        with temporary_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.fieldnames
            )

            writer.writeheader()

            # Preserve old values and leave new columns blank.
            for old_row in existing_rows:

                upgraded_row = {
                    field: old_row.get(field, "")
                    for field in self.fieldnames
                }

                writer.writerow(upgraded_row)

        # Replace the original only after migration succeeds.
        temporary_file.replace(self.alert_file)

        print("Existing security alert history upgraded.")

    # =====================================================
    # HELPER: CALCULATE TRAFFIC RATES
    # =====================================================

    @staticmethod
    def _get_traffic_rates(flow):

        duration = float(
            flow.get("duration_seconds", 0) or 0
        )

        packet_rate = flow.get("packet_rate")

        byte_rate = flow.get("byte_rate")

        if packet_rate is None:

            packet_rate = (
                float(flow.get("packet_count", 0)) / duration
                if duration > 0 else 0
            )

        if byte_rate is None:

            byte_rate = (
                float(flow.get("total_bytes", 0)) / duration
                if duration > 0 else 0
            )

        return packet_rate, byte_rate

    # =====================================================
    # HELPER: APPEND ALERT TO CSV
    # =====================================================

    def _append_alert(self, alert):

        write_header = (
            not self.alert_file.exists()
            or self.alert_file.stat().st_size == 0
        )

        # Ensure every row matches the current CSV schema.
        row = {
            field: alert.get(field, "")
            for field in self.fieldnames
        }

        with self.alert_file.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.fieldnames
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        return True

    # =====================================================
    # EXISTING: RECORD ISOLATION FOREST ANOMALY
    # =====================================================

    def record_detection(
        self,
        flow,
        detection_result,
        threat_result=None
    ):
        """
        Preserve the original Isolation Forest alert method.

        Normal flows are ignored.
        Anomalous flows are recorded.
        """

        if not detection_result["is_anomaly"]:
            return False

        threat_result = threat_result or {}

        packet_rate, byte_rate = (
            self._get_traffic_rates(flow)
        )

        alert = {

            "alert_time": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),

            "source_ip": flow["source_ip"],
            "source_port": flow["source_port"],

            "destination_ip": flow["destination_ip"],
            "destination_port": flow["destination_port"],

            "protocol": flow["protocol"],

            "duration_seconds": flow["duration_seconds"],
            "packet_count": flow["packet_count"],
            "total_bytes": flow["total_bytes"],

            "packet_rate": packet_rate,
            "byte_rate": byte_rate,

            "anomaly_score": detection_result.get(
                "anomaly_score", ""
            ),

            "decision_score": detection_result.get(
                "decision_score", ""
            ),

            "raw_score": detection_result.get(
                "raw_score", ""
            ),

            "threat_score": threat_result.get(
                "threat_score", ""
            ),

            "threat_level": threat_result.get(
                "threat_level", ""
            ),

            "score_reasons": json.dumps(
                threat_result.get("score_reasons", [])
            ),

            "baseline_percentiles": json.dumps(
                threat_result.get("baseline_percentiles", {})
            ),

            "status": "UNREVIEWED",
        }

        return self._append_alert(alert)

    # =====================================================
    # NEW: RECORD UNIFIED AI DETECTION
    # =====================================================

    def record_unified_detection(
        self,
        flow,
        unified_result,
        isolation_result=None,
        threat_result=None
    ):
        """
        Record an alert using the final unified decision.

        Supports:
        - Random Forest only detections.
        - Isolation Forest only detections.
        - Detections flagged by both models.

        Normal unified decisions are ignored.
        """

        # Do not create alerts for flows that neither model flagged.
        if not unified_result.get("is_attack_flagged", False):
            return False

        # Validate the required unified decision fields.
        required_fields = [
            "final_prediction",
            "evidence_status",
            "threat_score",
            "threat_level",
        ]

        missing_fields = [
            field for field in required_fields
            if field not in unified_result
        ]

        if missing_fields:
            raise ValueError(
                f"Unified result is missing fields: {missing_fields}"
            )

        isolation_result = isolation_result or {}
        threat_result = threat_result or {}

        packet_rate, byte_rate = (
            self._get_traffic_rates(flow)
        )

        alert = {

            "alert_time": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),

            "source_ip": flow["source_ip"],
            "source_port": flow["source_port"],

            "destination_ip": flow["destination_ip"],
            "destination_port": flow["destination_port"],

            "protocol": flow["protocol"],

            "duration_seconds": flow["duration_seconds"],
            "packet_count": flow["packet_count"],
            "total_bytes": flow["total_bytes"],

            "packet_rate": packet_rate,
            "byte_rate": byte_rate,

            # Preserve the original IF model scores.
            "anomaly_score": isolation_result.get(
                "anomaly_score", ""
            ),

            "decision_score": isolation_result.get(
                "decision_score", ""
            ),

            "raw_score": isolation_result.get(
                "raw_score", ""
            ),

            # Final unified classification.
            "final_prediction": unified_result["final_prediction"],

            "evidence_status": unified_result["evidence_status"],

            # Random Forest evidence.
            "rf_prediction": unified_result.get(
                "rf_prediction", "NO_RF_WINDOW"
            ),

            "rf_window_count": unified_result.get(
                "rf_window_count", 0
            ),

            "rf_flagged_window_count": unified_result.get(
                "rf_flagged_window_count", 0
            ),

            # Isolation Forest evidence.
            "if_prediction": isolation_result.get(
                "prediction", "UNKNOWN"
            ),

            "if_is_anomaly": isolation_result.get(
                "is_anomaly", ""
            ),

            "is_attack_flagged": True,

            # Unified threat priority.
            "threat_score": unified_result["threat_score"],

            "threat_level": unified_result["threat_level"],

            "score_reasons": json.dumps(
                unified_result.get("score_reasons", [])
            ),

            "baseline_percentiles": json.dumps(
                threat_result.get("baseline_percentiles", {})
            ),

            "status": "UNREVIEWED",
        }

        return self._append_alert(alert)
