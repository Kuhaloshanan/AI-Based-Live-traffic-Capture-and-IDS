
"""
Unified AI Decision Engine.

Combines:
1. Random Forest packet-window classifications.
2. Isolation Forest completed-flow anomaly detection.
3. Existing application-defined threat scores.

Produces one final verdict for each completed network flow.

IMPORTANT:
The combined scoring rules are application-defined heuristics.
They are not calibrated attack probabilities.
"""

import math
from collections import Counter


class UnifiedDecisionEngine:

    # Application-defined score floors.
    # These are policy choices, not scientifically calibrated values.
    RF_ONLY_SCORE_FLOOR = 40
    CORROBORATED_SCORE_FLOOR = 70

    def __init__(self):

        # Store RF window predictions by their shared flow identifier.
        self.rf_predictions_by_flow = {}

    # =====================================================
    # REGISTER RANDOM FOREST PREDICTION
    # =====================================================

    def add_rf_prediction(
        self,
        window_result,
        prediction_result
    ):
        """
        Store one Random Forest prediction under its flow key.

        Multiple packet windows can belong to one network flow.
        All their predictions are collected until that flow
        is finalized by Isolation Forest.
        """

        flow_key = window_result.get("flow_key")

        if flow_key is None:
            raise ValueError(
                "Random Forest window is missing flow_key."
            )

        # Verify that the flow identifier can be used as a dictionary key.
        try:
            hash(flow_key)
        except TypeError as error:
            raise ValueError(
                "Random Forest flow_key must be hashable."
            ) from error

        window_size = window_result.get("window_size")

        if window_size not in (10, 100):
            raise ValueError(
                "Unsupported Random Forest window size."
            )

        prediction_window_size = prediction_result.get(
            "window_size"
        )

        if prediction_window_size != window_size:
            raise ValueError(
                "Prediction window size does not match input window."
            )

        prediction = prediction_result.get("prediction")

        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError(
                "Random Forest prediction is missing or invalid."
            )

        is_benign = prediction_result.get("is_benign")

        if not isinstance(is_benign, bool):
            raise ValueError(
                "Random Forest is_benign must be a boolean."
            )

        if is_benign != (prediction == "BENIGN"):
            raise ValueError(
                "Random Forest prediction and is_benign disagree."
            )

        model_score = float(
            prediction_result.get("model_score")
        )

        if (
            not math.isfinite(model_score)
            or not 0 <= model_score <= 1
        ):
            raise ValueError(
                "Random Forest model_score must be between 0 and 1."
            )

        # Prepare the prediction record.
        record = {
            "window_size": window_size,
            "prediction": prediction,
            "is_benign": is_benign,
            "model_score": model_score
        }

        # Create a prediction list for a new flow.
        if flow_key not in self.rf_predictions_by_flow:

            self.rf_predictions_by_flow[flow_key] = []

        # Append the new prediction to the correct flow.
        self.rf_predictions_by_flow[flow_key].append(record)

    # =====================================================
    # DETERMINE FINAL THREAT LEVEL
    # =====================================================

    @staticmethod
    def get_threat_level(score):
        """
        Convert the application-defined score into a threat level.
        """

        if score < 25:
            return "LOW"

        elif score < 50:
            return "MEDIUM"

        elif score < 75:
            return "HIGH"

        return "CRITICAL"

    # =====================================================
    # FINALIZE ONE NETWORK FLOW
    # =====================================================

    def finalize_flow(
        self,
        flow,
        isolation_result,
        threat_result
    ):
        """
        Combine RF and IF evidence into one final flow verdict.

        RF predictions are retrieved using flow_key and removed
        after finalization to prevent reuse by another flow.
        """

        flow_key = flow.get("flow_key")

        if flow_key is None:
            raise ValueError(
                "Completed network flow is missing flow_key."
            )

        # Retrieve all RF predictions associated with this flow.
        # pop() also removes them after retrieval.
        rf_windows = self.rf_predictions_by_flow.pop(
            flow_key,
            []
        )

        # -------------------------------------------------
        # ISOLATION FOREST RESULT
        # -------------------------------------------------

        if_anomaly = bool(
            isolation_result.get("is_anomaly", False)
        )

        if_prediction = isolation_result.get(
            "prediction",
            "UNKNOWN"
        )

        if_threat_score = float(
            threat_result.get("threat_score", 0)
        )

        if (
            not math.isfinite(if_threat_score)
            or not 0 <= if_threat_score <= 100
        ):
            raise ValueError(
                "Isolation Forest threat score must be between 0 and 100."
            )

        # -------------------------------------------------
        # RANDOM FOREST RESULT AGGREGATION
        # -------------------------------------------------

        rf_benign_count = sum(
            1 for item in rf_windows
            if item["is_benign"]
        )

        rf_attack_windows = [
            item for item in rf_windows
            if not item["is_benign"]
        ]

        rf_attack_count = len(rf_attack_windows)

        rf_category_counts = Counter(
            item["prediction"]
            for item in rf_attack_windows
        )

        # Select the most frequently predicted non-BENIGN category.
        if rf_category_counts:

            rf_prediction = (
                rf_category_counts.most_common(1)[0][0]
            )

        elif rf_windows:

            rf_prediction = "BENIGN"

        else:

            rf_prediction = "NO_RF_WINDOW"

        rf_flagged = rf_attack_count > 0

        # -------------------------------------------------
        # UNIFIED FINAL PREDICTION
        # -------------------------------------------------

        if rf_flagged and if_anomaly:

            # Both models have raised a detection signal.
            final_prediction = rf_prediction

            evidence_status = "BOTH_MODELS_FLAGGED"

        elif rf_flagged:

            # RF classified one or more windows as non-BENIGN,
            # but Isolation Forest did not detect an anomaly.
            final_prediction = rf_prediction

            evidence_status = "RANDOM_FOREST_ONLY"

        elif if_anomaly:

            # IF detected unusual behaviour, but RF did not
            # produce a non-BENIGN classification.
            final_prediction = "SUSPICIOUS"

            if rf_windows:

                evidence_status = (
                    "ISOLATION_FOREST_FLAGGED_RF_BENIGN"
                )

            else:

                evidence_status = (
                    "ISOLATION_FOREST_ONLY_NO_RF_WINDOW"
                )

        else:

            final_prediction = "BENIGN"

            if rf_windows:

                evidence_status = "BOTH_MODELS_BENIGN"

            else:

                evidence_status = (
                    "ISOLATION_FOREST_ONLY_NO_RF_WINDOW"
                )

        # -------------------------------------------------
        # UNIFIED APPLICATION-DEFINED THREAT SCORE
        # -------------------------------------------------

        if rf_flagged and if_anomaly:

            unified_score = max(
                if_threat_score,
                self.CORROBORATED_SCORE_FLOOR
            )

        elif rf_flagged:

            unified_score = max(
                if_threat_score,
                self.RF_ONLY_SCORE_FLOOR
            )

        elif if_anomaly:

            unified_score = if_threat_score

        else:

            unified_score = 0

        unified_score = min(
            100,
            round(unified_score, 2)
        )

        threat_level = self.get_threat_level(
            unified_score
        )

        # -------------------------------------------------
        # EXPLAIN THE FINAL DECISION
        # -------------------------------------------------

        score_reasons = list(
            threat_result.get("score_reasons", [])
        )

        if rf_flagged:

            score_reasons.append(
                f"Random Forest flagged {rf_attack_count} "
                f"out of {len(rf_windows)} packet windows."
            )

            score_reasons.append(
                f"Most frequent RF attack category: "
                f"{rf_prediction}."
            )

        elif rf_windows:

            score_reasons.append(
                "All completed Random Forest windows "
                "were classified as BENIGN."
            )

        else:

            score_reasons.append(
                "No completed Random Forest window was available "
                "for this flow."
            )

        if if_anomaly:

            score_reasons.append(
                "Isolation Forest classified the completed "
                "network flow as ANOMALY."
            )

        else:

            score_reasons.append(
                "Isolation Forest classified the completed "
                "network flow as NORMAL."
            )

        # -------------------------------------------------
        # RETURN ONE UNIFIED RESULT
        # -------------------------------------------------

        return {

            "flow_key": flow_key,

            "final_prediction": final_prediction,

            "threat_score": unified_score,

            "threat_level": threat_level,

            "evidence_status": evidence_status,

            "is_attack_flagged": (
                rf_flagged or if_anomaly
            ),

            # Random Forest evidence.
            "rf_prediction": rf_prediction,

            "rf_window_count": len(rf_windows),

            "rf_benign_window_count": rf_benign_count,

            "rf_flagged_window_count": rf_attack_count,

            "rf_category_counts": dict(
                rf_category_counts
            ),

            "rf_window_predictions": rf_windows,

            # Isolation Forest evidence.
            "if_prediction": if_prediction,

            "if_is_anomaly": if_anomaly,

            "if_threat_score": if_threat_score,

            # Explanation.
            "score_reasons": score_reasons,

            "classification_note": (
                "Model classifications and anomaly detections "
                "are evidence signals, not independently "
                "confirmed attacks. Threat scores are "
                "application-defined and not calibrated probabilities."
            )
        }
