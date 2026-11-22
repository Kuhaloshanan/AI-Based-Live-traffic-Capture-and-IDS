"""
AI-Based Live Network Intrusion Detection Monitor.

Integrates:
1. Scapy live packet capture
2. Random Forest packet-window classification
3. Isolation Forest network-flow anomaly detection
4. Unified decision engine combining RF and IF evidence
5. Application-defined threat scoring
6. Persistent security alert recording
7. Persistent RF, flow, and unified decision telemetry

Important design note:
Random Forest classifies completed packet windows, while Isolation
Forest evaluates completed network flows. RF window predictions are
associated with a flow through the shared flow_key. When the flow is
completed, the unified engine combines the available evidence into one
flow-level decision.
"""

import time
from queue import Queue, Empty

from scapy.all import AsyncSniffer

from src.flow_processor import FlowProcessor
from src.live_detection_engine import LiveDetectionEngine
from src.alert_manager import AlertManager
from src.threat_scoring import ThreatScorer
from src.telemetry_logger import TelemetryLogger
from src.unified_decision_engine import UnifiedDecisionEngine
from src.unified_telemetry_logger import UnifiedTelemetryLogger

from src.live_rf_window_processor import LiveRFWindowProcessor
from src.live_rf_prediction_engine import LiveRandomForestEngine
from src.live_rf_telemetry_logger import LiveRFTelemetryLogger


# =========================================================
# CONFIGURATION
# =========================================================

INTERFACE_NAME = "Wi-Fi"
FLOW_IDLE_TIMEOUT = 30

packet_queue = Queue()

flow_processor = FlowProcessor(
    idle_timeout=FLOW_IDLE_TIMEOUT
)

# Session statistics for the two detection models and the
# unified decision pipeline.
MONITORING_STATS = {
    "total_flows": 0,
    "normal_flows": 0,
    "anomalous_flows": 0,
    "detection_errors": 0,

    "total_rf_windows": 0,
    "rf_benign_windows": 0,
    "rf_flagged_windows": 0,
    "rf_errors": 0,
    "rf_telemetry_errors": 0,

    "unified_decisions": 0,
    "unified_flagged_flows": 0,
    "unified_errors": 0,
    "unified_telemetry_errors": 0,
    "unified_alert_errors": 0,
}


def reset_monitoring_stats():
    """Reset counters when a new monitoring session starts."""
    for key in MONITORING_STATS:
        MONITORING_STATS[key] = 0


# =========================================================
# RANDOM FOREST PACKET-WINDOW DETECTION
# =========================================================

def display_completed_rf_window(
    window_result,
    rf_engine,
    rf_telemetry_logger,
    unified_engine,
):
    """
    Classify a completed packet window using Random Forest.

    Store the successful prediction in the unified engine. The engine
    groups predictions by flow_key until Isolation Forest completes
    the corresponding network flow.
    """

    MONITORING_STATS["total_rf_windows"] += 1

    try:
        # STEP 1: Generate the Random Forest prediction.
        result = rf_engine.predict_window(window_result)

        # STEP 2: Associate this window prediction with its network flow.
        # A failure here must not prevent RF telemetry from being saved.
        try:
            unified_engine.add_rf_prediction(
                window_result,
                result,
            )
            print("RF prediction linked to the unified decision engine.")

        except Exception as unified_error:
            MONITORING_STATS["unified_errors"] += 1
            print("Could not link RF prediction to unified engine.")
            print(f"Error: {unified_error}")

        # STEP 3: Save the original RF prediction and its features.
        try:
            rf_telemetry_logger.record_prediction(
                window_result,
                result,
            )

        except Exception as telemetry_error:
            MONITORING_STATS["rf_telemetry_errors"] += 1
            print("Could not save Random Forest telemetry.")
            print(f"Error: {telemetry_error}")

        # STEP 4: Update RF statistics.
        prediction = result["prediction"]

        if result["is_benign"]:
            MONITORING_STATS["rf_benign_windows"] += 1
        else:
            MONITORING_STATS["rf_flagged_windows"] += 1

        # STEP 5: Display the window-level model result.
        print("\n" + "=" * 70)
        print("RANDOM FOREST PACKET-WINDOW CLASSIFICATION")
        print("=" * 70)

        print(f"Window size:        {result['window_size']} packets")
        print(f"Predicted category: {prediction}")
        print(f"Model score:        {result['model_score']:.4f}")

        print("\nTop three model predictions:")
        for item in result["top_predictions"]:
            print(
                f"  {item['category']}: "
                f"{item['model_score']:.4f}"
            )

        if result["is_benign"]:
            print("\nClassification result: BENIGN")
        else:
            print(
                "\nMODEL FLAG: "
                f"Traffic classified as {prediction}"
            )
            print(
                "Note: This is a model classification, "
                "not independently confirmed attack activity."
            )

        print("=" * 70)

    except Exception as error:
        MONITORING_STATS["rf_errors"] += 1
        print("\nRandom Forest prediction failed.")
        print(f"Error: {error}")


# =========================================================
# UNIFIED FLOW DECISION DISPLAY AND RECORDING
# =========================================================

def display_unified_decision(
    flow,
    ai_result,
    threat_result,
    unified_engine,
    unified_telemetry_logger,
    alert_manager,
):
    """
    Combine RF window predictions and the IF completed-flow result.

    The unified engine produces one final decision per completed flow.
    The decision is written to unified telemetry, and only flagged
    decisions are persisted as security alerts by AlertManager.
    """

    try:
        # STEP 1: Combine all RF windows associated with this flow
        # with the Isolation Forest result for the completed flow.
        unified_result = unified_engine.finalize_flow(
            flow,
            ai_result,
            threat_result,
        )

        MONITORING_STATS["unified_decisions"] += 1

        if unified_result["is_attack_flagged"]:
            MONITORING_STATS["unified_flagged_flows"] += 1

        # STEP 2: Display the final flow-level decision.
        print("\n" + "=" * 70)
        print("UNIFIED AI NETWORK THREAT DECISION")
        print("=" * 70)

        print(
            f"Final prediction:       "
            f"{unified_result['final_prediction']}"
        )
        print(
            f"Unified threat score:   "
            f"{unified_result['threat_score']}/100"
        )
        print(
            f"Unified threat level:   "
            f"{unified_result['threat_level']}"
        )
        print(
            f"Evidence status:        "
            f"{unified_result['evidence_status']}"
        )
        print(
            f"RF aggregate prediction: "
            f"{unified_result['rf_prediction']}"
        )
        print(
            f"RF windows received:    "
            f"{unified_result['rf_window_count']}"
        )
        print(
            f"RF benign windows:      "
            f"{unified_result['rf_benign_window_count']}"
        )
        print(
            f"RF flagged windows:     "
            f"{unified_result['rf_flagged_window_count']}"
        )
        print(
            f"Isolation Forest:       "
            f"{unified_result['if_prediction']}"
        )
        print(
            f"IF anomaly detected:    "
            f"{unified_result['if_is_anomaly']}"
        )
        print(
            f"Attack flagged:         "
            f"{unified_result['is_attack_flagged']}"
        )

        print("\nUnified score reasons:")
        for reason in unified_result["score_reasons"]:
            print(f"  - {reason}")

        print(
            "\nClassification note: "
            f"{unified_result['classification_note']}"
        )

        print(
            "Note: The unified threat score is an "
            "application-defined heuristic, not a calibrated probability."
        )
        print("=" * 70)

        # STEP 3: Save one unified decision row for this completed flow.
        try:
            unified_telemetry_logger.record_decision(
                flow,
                unified_result,
                ai_result,
                threat_result,
            )
        except Exception as telemetry_error:
            MONITORING_STATS["unified_telemetry_errors"] += 1
            print("Could not save unified decision telemetry.")
            print(f"Error: {telemetry_error}")

        # STEP 4: Record an alert only when the unified engine flags it.
        try:
            recorded = alert_manager.record_unified_detection(
                flow,
                unified_result,
                ai_result,
                threat_result,
            )

            if recorded:
                print("\nUNIFIED SECURITY ALERT RECORDED")
                print(f"Alert file: {alert_manager.alert_file}")
            elif unified_result["is_attack_flagged"]:
                print(
                    "\nThe flow was flagged, but the alert manager "
                    "did not confirm that an alert was written."
                )
            else:
                print("\nNo security alert generated for this flow.")

        except Exception as alert_error:
            MONITORING_STATS["unified_alert_errors"] += 1
            print("Could not save unified security alert.")
            print(f"Error: {alert_error}")

    except Exception as unified_error:
        MONITORING_STATS["unified_errors"] += 1
        print("Unified flow decision failed.")
        print(f"Error: {unified_error}")
        print(
            "The Isolation Forest result and legacy flow telemetry "
            "were processed separately where possible."
        )


# =========================================================
# ISOLATION FOREST FLOW DETECTION
# =========================================================

def display_completed_flow(
    flow,
    detection_engine,
    alert_manager,
    threat_scorer,
    telemetry_logger,
    unified_engine,
    unified_telemetry_logger,
):
    """
    Process one completed network flow.

    Performs Isolation Forest detection, threat scoring, legacy flow
    telemetry, unified RF/IF decision-making, unified telemetry, and
    security alert recording.
    """

    print("\n" + "=" * 70)
    print("COMPLETED NETWORK FLOW")
    print("=" * 70)

    print(f"Source:       {flow['source_ip']}:{flow['source_port']}")
    print(
        f"Destination:  "
        f"{flow['destination_ip']}:{flow['destination_port']}"
    )
    print(f"Protocol:     {flow['protocol']}")
    print(f"Duration:     {flow['duration_seconds']:.4f} seconds")
    print(f"Packets:      {flow['packet_count']}")
    print(f"Total bytes:  {flow['total_bytes']}")
    print(f"Forward packets: {flow['forward_packets']}")
    print(f"Reverse packets: {flow['reverse_packets']}")

    print("-" * 70)
    print("ISOLATION FOREST FLOW DETECTION")
    print("-" * 70)

    MONITORING_STATS["total_flows"] += 1

    # STEP 1: Run Isolation Forest and calculate the IF-based score.
    try:
        ai_result = detection_engine.predict_flow(flow)

        threat_result = threat_scorer.score_flow(
            flow,
            ai_result,
        )

    except Exception as error:
        MONITORING_STATS["detection_errors"] += 1
        print("Isolation Forest detection or threat scoring failed.")
        print(f"Error: {error}")

        # Keep an error record in the legacy telemetry file.
        try:
            telemetry_logger.record_flow(
                flow,
                detection_status="ERROR",
            )
        except Exception as telemetry_error:
            print("Could not save error flow telemetry.")
            print(f"Error: {telemetry_error}")

        # Do not label the IF result as BENIGN when detection failed.
        # The unified engine requires a valid IF result, so no unified
        # verdict is generated for this flow in this error path.
        # Remove any RF windows already waiting for this completed flow;
        # otherwise, a later flow reusing the same 5-tuple could inherit
        # stale RF evidence.
        flow_key = flow.get("flow_key")
        pending_rf = getattr(
            unified_engine,
            "rf_predictions_by_flow",
            None,
        )
        if flow_key is not None and isinstance(pending_rf, dict):
            pending_rf.pop(flow_key, None)

        print("Unified decision skipped because IF evidence is unavailable.")
        print("=" * 70)
        return

    # STEP 2: Save the original Isolation Forest flow telemetry.
    try:
        telemetry_logger.record_flow(
            flow,
            ai_result,
            threat_result,
        )
    except Exception as telemetry_error:
        print("Could not save flow telemetry.")
        print(f"Error: {telemetry_error}")

    # STEP 3: Update IF statistics.
    if ai_result["is_anomaly"]:
        MONITORING_STATS["anomalous_flows"] += 1
    else:
        MONITORING_STATS["normal_flows"] += 1

    # STEP 4: Display the Isolation Forest result.
    print(f"Threat score: {threat_result['threat_score']}/100")
    print(f"Threat level: {threat_result['threat_level']}")

    for reason in threat_result["score_reasons"]:
        print(f"  - {reason}")

    print(f"Detection status: {ai_result['prediction']}")
    print(f"Anomaly margin: {ai_result['anomaly_score']:.6f}")
    print(f"Decision score: {ai_result['decision_score']:.6f}")
    print(f"Raw model score: {ai_result['raw_score']:.6f}")

    # STEP 5: Combine IF flow evidence with RF window evidence.
    display_unified_decision(
        flow,
        ai_result,
        threat_result,
        unified_engine,
        unified_telemetry_logger,
        alert_manager,
    )

    print("=" * 70)


# =========================================================
# PROCESS CAPTURED PACKETS
# =========================================================

def process_queued_packet(
    packet,
    detection_engine,
    alert_manager,
    threat_scorer,
    telemetry_logger,
    rf_window_processor,
    rf_engine,
    rf_telemetry_logger,
    unified_engine,
    unified_telemetry_logger,
):
    """
    Process one captured packet through the shared flow and RF pipelines.

    The Isolation Forest flow processor runs first. It may complete an
    old flow when this packet arrives after the idle timeout. Finalizing
    that old flow before processing the packet in the RF processor
    prevents a new RF window with the same 5-tuple from being attached
    to the previous flow.
    """

    # -----------------------------------------------------
    # PATH A: UPDATE / COMPLETE ISOLATION FOREST NETWORK FLOWS
    # -----------------------------------------------------

    try:
        completed_flows = flow_processor.process_packet(packet)

        for flow in completed_flows:
            display_completed_flow(
                flow,
                detection_engine,
                alert_manager,
                threat_scorer,
                telemetry_logger,
                unified_engine,
                unified_telemetry_logger,
            )

    except Exception as flow_error:
        MONITORING_STATS["detection_errors"] += 1
        print("Isolation Forest flow processing error.")
        print(f"Error: {flow_error}")

    # -----------------------------------------------------
    # PATH B: RANDOM FOREST PACKET-WINDOW PROCESSING
    # -----------------------------------------------------

    try:
        completed_windows = rf_window_processor.process_packet(packet)

        for window_result in completed_windows:
            display_completed_rf_window(
                window_result,
                rf_engine,
                rf_telemetry_logger,
                unified_engine,
            )

    except Exception as rf_error:
        MONITORING_STATS["rf_errors"] += 1
        print("Random Forest packet processing error.")
        print(f"Error: {rf_error}")


# =========================================================
# START LIVE AI MONITORING
# =========================================================

def start_live_monitor():
    """Initialize all components and start live packet monitoring."""

    reset_monitoring_stats()

    print("=" * 70)
    print("AI CYBER THREAT INTELLIGENCE SYSTEM")
    print("LIVE AI-BASED NETWORK INTRUSION DETECTION")
    print("=" * 70)

    print(f"Interface: {INTERFACE_NAME}")
    print(f"Flow timeout: {FLOW_IDLE_TIMEOUT} seconds")

    # -----------------------------------------------------
    # INITIALIZE ISOLATION FOREST COMPONENTS
    # -----------------------------------------------------

    detection_engine = LiveDetectionEngine()
    alert_manager = AlertManager()
    threat_scorer = ThreatScorer()
    telemetry_logger = TelemetryLogger()

    # -----------------------------------------------------
    # INITIALIZE RANDOM FOREST COMPONENTS
    # -----------------------------------------------------

    print("\nInitializing Random Forest packet-window detection...")

    rf_window_processor = LiveRFWindowProcessor(
        window_sizes=(10, 100),
        idle_timeout=FLOW_IDLE_TIMEOUT,
    )

    rf_engine = LiveRandomForestEngine()
    rf_telemetry_logger = LiveRFTelemetryLogger()

    # -----------------------------------------------------
    # INITIALIZE UNIFIED DECISION COMPONENTS
    # -----------------------------------------------------

    unified_engine = UnifiedDecisionEngine()
    unified_telemetry_logger = UnifiedTelemetryLogger()

    print("Unified AI decision engine initialized.")
    print(
        f"Unified decision history: "
        f"{unified_telemetry_logger.telemetry_file}"
    )

    print("Random Forest packet-window detection initialized.")
    print("\nAll detection components initialized.")

    print(f"Alert history: {alert_manager.alert_file}")
    print(f"Flow telemetry: {telemetry_logger.telemetry_file}")
    print(f"RF prediction telemetry: {rf_telemetry_logger.log_file}")

    # -----------------------------------------------------
    # CONFIGURE SCAPY PACKET CAPTURE
    # -----------------------------------------------------

    sniffer = AsyncSniffer(
        iface=INTERFACE_NAME,
        prn=packet_queue.put,
        store=False,
    )

    try:
        # Start live packet capture.
        sniffer.start()

        print("\nPacket capture started successfully.")
        print("Random Forest packet-window classification is active.")
        print("Isolation Forest flow anomaly detection is active.")
        print("Unified RF + IF decisions are active.")
        print("Threat scoring and security alert recording are active.")
        print("Flow and unified decision telemetry are active.")
        print("Press CTRL + C to stop monitoring.\n")

        # -------------------------------------------------
        # MAIN MONITORING LOOP
        # -------------------------------------------------

        while True:
            try:
                packet = packet_queue.get(timeout=1)

                process_queued_packet(
                    packet,
                    detection_engine,
                    alert_manager,
                    threat_scorer,
                    telemetry_logger,
                    rf_window_processor,
                    rf_engine,
                    rf_telemetry_logger,
                    unified_engine,
                    unified_telemetry_logger,
                )

            except Empty:
                pass

            # Finalize flows that have been inactive for the timeout.
            completed_flows = flow_processor.expire_flows(time.time())

            for flow in completed_flows:
                display_completed_flow(
                    flow,
                    detection_engine,
                    alert_manager,
                    threat_scorer,
                    telemetry_logger,
                    unified_engine,
                    unified_telemetry_logger,
                )

    except KeyboardInterrupt:
        print("\nStopping live monitoring...")

    finally:
        # -------------------------------------------------
        # STOP PACKET CAPTURE
        # -------------------------------------------------

        if sniffer.running:
            sniffer.stop()

        # -------------------------------------------------
        # PROCESS REMAINING QUEUED PACKETS
        # -------------------------------------------------

        while True:
            try:
                packet = packet_queue.get_nowait()

                process_queued_packet(
                    packet,
                    detection_engine,
                    alert_manager,
                    threat_scorer,
                    telemetry_logger,
                    rf_window_processor,
                    rf_engine,
                    rf_telemetry_logger,
                    unified_engine,
                    unified_telemetry_logger,
                )

            except Empty:
                break

        # -------------------------------------------------
        # FINALIZE REMAINING NETWORK FLOWS
        # -------------------------------------------------

        remaining_flows = flow_processor.flush()

        for flow in remaining_flows:
            display_completed_flow(
                flow,
                detection_engine,
                alert_manager,
                threat_scorer,
                telemetry_logger,
                unified_engine,
                unified_telemetry_logger,
            )

        # -------------------------------------------------
        # DISPLAY SESSION STATISTICS
        # -------------------------------------------------

        print("\nLive AI monitoring stopped.")
        print("All remaining network flows have been processed.")

        print("\n" + "=" * 65)
        print("LIVE MONITORING SESSION SUMMARY")
        print("=" * 65)

        print("\nISOLATION FOREST RESULTS")
        total = MONITORING_STATS["total_flows"]
        anomalies = MONITORING_STATS["anomalous_flows"]

        print(f"Total flows inspected: {total}")
        print(f"Normal flows:          {MONITORING_STATS['normal_flows']}")
        print(f"Anomalous flows:       {anomalies}")
        print(f"Detection errors:      {MONITORING_STATS['detection_errors']}")

        if total > 0:
            anomaly_rate = (anomalies / total) * 100
            print(f"Anomaly rate:          {anomaly_rate:.2f}%")

        print("\nRANDOM FOREST RESULTS")
        print(f"Total packet windows:  {MONITORING_STATS['total_rf_windows']}")
        print(f"BENIGN windows:        {MONITORING_STATS['rf_benign_windows']}")
        print(f"Non-BENIGN predictions: {MONITORING_STATS['rf_flagged_windows']}")
        print(f"RF processing errors:  {MONITORING_STATS['rf_errors']}")
        print(f"RF telemetry errors:   {MONITORING_STATS['rf_telemetry_errors']}")

        print("\nUNIFIED DECISION RESULTS")
        print(f"Unified decisions:     {MONITORING_STATS['unified_decisions']}")
        print(f"Flagged flows:         {MONITORING_STATS['unified_flagged_flows']}")
        print(f"Unified errors:        {MONITORING_STATS['unified_errors']}")
        print(
            "Unified telemetry errors: "
            f"{MONITORING_STATS['unified_telemetry_errors']}"
        )
        print(
            "Unified alert errors:      "
            f"{MONITORING_STATS['unified_alert_errors']}"
        )

        print("=" * 65)


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    start_live_monitor()
