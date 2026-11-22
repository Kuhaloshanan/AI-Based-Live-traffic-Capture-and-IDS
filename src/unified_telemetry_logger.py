
"""
Unified Telemetry Logger.

Records one combined AI decision per completed network flow.

Preserves:
1. Random Forest packet-window evidence.
2. Isolation Forest flow-anomaly evidence.
3. Unified threat classification and scoring.
4. Explainable decision information.

This logger uses a separate CSV file so existing
Random Forest and Isolation Forest telemetry remains intact.
"""

import csv
import json

from datetime import datetime
from pathlib import Path


# Project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Destination for unified flow decisions.
UNIFIED_DECISIONS_FILE = (
    PROJECT_ROOT / "logs" / "unified_decisions.csv"
)


# CSV column definitions.
UNIFIED_TELEMETRY_COLUMNS = [
    "event_time",
    "flow_key",
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
    "final_prediction",
    "threat_score",
    "threat_level",
    "evidence_status",
    "is_attack_flagged",
    "rf_prediction",
    "rf_window_count",
    "rf_benign_window_count",
    "rf_flagged_window_count",
    "rf_category_counts",
    "rf_window_predictions",
    "if_prediction",
    "if_is_anomaly",
    "if_anomaly_score",
    "if_decision_score",
    "if_raw_score",
    "if_threat_score",
    "score_reasons",
    "baseline_percentiles",
    "classification_note",
]


class UnifiedTelemetryLogger:
    """
    Persist one unified decision for each completed network flow.
    """

    def __init__(
        self,
        telemetry_file=UNIFIED_DECISIONS_FILE
    ):

        self.telemetry_file = Path(telemetry_file)

        # Ensure the logs directory exists.
        self.telemetry_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Avoid appending records to an incompatible CSV schema.
        if (
            self.telemetry_file.exists()
            and self.telemetry_file.stat().st_size > 0
        ):

            with self.telemetry_file.open(
                "r",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                existing_header = next(
                    csv.reader(csv_file),
                    None
                )

            if existing_header != UNIFIED_TELEMETRY_COLUMNS:
                raise ValueError(
                    "Unified telemetry CSV schema mismatch. "
                    "Back up or rename the existing file before "
                    "using a different schema."
                )

    # =====================================================
    # RECORD ONE UNIFIED FLOW DECISION
    # =====================================================

    def record_decision(
        self,
        flow,
        unified_result,
        isolation_result=None,
        threat_result=None
    ):
        """
        Append one unified AI decision to the CSV.

        Parameters:
            flow:
                Completed network flow from FlowProcessor.

            unified_result:
                Final decision from UnifiedDecisionEngine.

            isolation_result:
                Original Isolation Forest prediction result.

            threat_result:
                Original ThreatScorer result.
        """

        isolation_result = isolation_result or {}
        threat_result = threat_result or {}

        # Required fields protect against incomplete decisions.
        required_fields = [
            "flow_key",
            "final_prediction",
            "threat_score",
            "threat_level",
            "evidence_status",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in unified_result
        ]

        if missing_fields:
            raise ValueError(
                f"Unified result is missing fields: {missing_fields}"
            )

        # Verify that the decision belongs to the supplied flow.
        if "flow_key" not in flow:
            raise ValueError(
                "Completed flow is missing flow_key."
            )

        if flow["flow_key"] != unified_result["flow_key"]:
            raise ValueError(
                "Flow key mismatch between flow and unified result."
            )

        # Prepare one persistent telemetry record.
        row = {

            "event_time": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),

            # JSON representation keeps the tuple structure readable.
            "flow_key": json.dumps(flow["flow_key"]),

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

            # Final unified decision.
            "final_prediction": unified_result["final_prediction"],

            "threat_score": unified_result["threat_score"],
            "threat_level": unified_result["threat_level"],

            "evidence_status": unified_result["evidence_status"],

            "is_attack_flagged": unified_result.get(
                "is_attack_flagged", False
            ),

            # Random Forest evidence.
            "rf_prediction": unified_result.get(
                "rf_prediction", "NO_RF_WINDOW"
            ),

            "rf_window_count": unified_result.get(
                "rf_window_count", 0
            ),

            "rf_benign_window_count": unified_result.get(
                "rf_benign_window_count", 0
            ),

            "rf_flagged_window_count": unified_result.get(
                "rf_flagged_window_count", 0
            ),

            "rf_category_counts": json.dumps(
                unified_result.get("rf_category_counts", {}),
                sort_keys=True
            ),

            "rf_window_predictions": json.dumps(
                unified_result.get("rf_window_predictions", []),
                sort_keys=True
            ),

            # Isolation Forest evidence.
            "if_prediction": isolation_result.get(
                "prediction", "UNKNOWN"
            ),

            "if_is_anomaly": isolation_result.get(
                "is_anomaly", ""
            ),

            "if_anomaly_score": isolation_result.get(
                "anomaly_score", ""
            ),

            "if_decision_score": isolation_result.get(
                "decision_score", ""
            ),

            "if_raw_score": isolation_result.get(
                "raw_score", ""
            ),

            "if_threat_score": unified_result.get(
                "if_threat_score", ""
            ),

            # Explainability and baseline evidence.
            "score_reasons": json.dumps(
                unified_result.get("score_reasons", []),
                sort_keys=True
            ),

            "baseline_percentiles": json.dumps(
                threat_result.get("baseline_percentiles", {}),
                sort_keys=True
            ),

            "classification_note": unified_result.get(
                "classification_note", ""
            ),
        }

        # Determine whether a header must be written.
        write_header = (
            not self.telemetry_file.exists()
            or self.telemetry_file.stat().st_size == 0
        )

        # Append the decision without overwriting previous records.
        with self.telemetry_file.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=UNIFIED_TELEMETRY_COLUMNS
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        return True
