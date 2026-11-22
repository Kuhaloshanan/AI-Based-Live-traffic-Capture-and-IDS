
import unittest

from src.unified_decision_engine import UnifiedDecisionEngine


class TestUnifiedDecisionEngine(unittest.TestCase):

    def setUp(self):
        """Create a fresh decision engine for every test."""
        self.engine = UnifiedDecisionEngine()

        self.flow_key = (
            "TCP",
            ("192.168.1.10", 50000),
            ("1.1.1.1", 443)
        )

        self.flow = {
            "flow_key": self.flow_key
        }

    def add_rf_result(
        self,
        prediction="BENIGN",
        window_size=10,
        model_score=0.95
    ):
        """Add a simulated Random Forest window prediction."""

        window = {
            "flow_key": self.flow_key,
            "window_size": window_size
        }

        prediction_result = {
            "window_size": window_size,
            "prediction": prediction,
            "is_benign": prediction == "BENIGN",
            "model_score": model_score
        }

        self.engine.add_rf_prediction(
            window,
            prediction_result
        )

    def finalize(
        self,
        is_anomaly=False,
        threat_score=0
    ):
        """Finalize a simulated network flow."""

        isolation_result = {
            "is_anomaly": is_anomaly,
            "prediction": (
                "ANOMALY" if is_anomaly else "NORMAL"
            )
        }

        threat_result = {
            "threat_score": threat_score,
            "score_reasons": []
        }

        return self.engine.finalize_flow(
            self.flow,
            isolation_result,
            threat_result
        )

    # -------------------------------------------------
    # TEST 1: BOTH MODELS CLASSIFY TRAFFIC AS NORMAL
    # -------------------------------------------------

    def test_both_models_benign(self):

        self.add_rf_result("BENIGN")

        result = self.finalize(
            is_anomaly=False,
            threat_score=0
        )

        self.assertEqual(
            result["final_prediction"],
            "BENIGN"
        )

        self.assertEqual(result["threat_score"], 0)

        self.assertEqual(
            result["evidence_status"],
            "BOTH_MODELS_BENIGN"
        )

    # -------------------------------------------------
    # TEST 2: ONLY RANDOM FOREST FLAGS AN ATTACK
    # -------------------------------------------------

    def test_random_forest_only_flags_attack(self):

        self.add_rf_result("DDoS")

        result = self.finalize(
            is_anomaly=False,
            threat_score=0
        )

        self.assertEqual(
            result["final_prediction"],
            "DDoS"
        )

        self.assertEqual(result["threat_score"], 40)

        self.assertEqual(
            result["evidence_status"],
            "RANDOM_FOREST_ONLY"
        )

    # -------------------------------------------------
    # TEST 3: BOTH MODELS FLAG THE FLOW
    # -------------------------------------------------

    def test_both_models_flag_attack(self):

        self.add_rf_result("DDoS")

        result = self.finalize(
            is_anomaly=True,
            threat_score=35
        )

        self.assertEqual(
            result["final_prediction"],
            "DDoS"
        )

        self.assertEqual(result["threat_score"], 70)

        self.assertEqual(
            result["evidence_status"],
            "BOTH_MODELS_FLAGGED"
        )

        self.assertEqual(result["threat_level"], "HIGH")

    # -------------------------------------------------
    # TEST 4: ONLY ISOLATION FOREST FLAGS THE FLOW
    # -------------------------------------------------

    def test_isolation_forest_only_flags_anomaly(self):

        self.add_rf_result("BENIGN")

        result = self.finalize(
            is_anomaly=True,
            threat_score=55
        )

        self.assertEqual(
            result["final_prediction"],
            "SUSPICIOUS"
        )

        self.assertEqual(result["threat_score"], 55)

        self.assertEqual(
            result["evidence_status"],
            "ISOLATION_FOREST_FLAGGED_RF_BENIGN"
        )

    # -------------------------------------------------
    # TEST 5: FLOW HAS NO COMPLETED RF WINDOW
    # -------------------------------------------------

    def test_flow_without_rf_window(self):

        result = self.finalize(
            is_anomaly=False,
            threat_score=0
        )

        self.assertEqual(
            result["final_prediction"],
            "BENIGN"
        )

        self.assertEqual(
            result["rf_prediction"],
            "NO_RF_WINDOW"
        )

        self.assertEqual(result["rf_window_count"], 0)

    # -------------------------------------------------
    # TEST 6: MULTIPLE RF WINDOWS ARE AGGREGATED
    # -------------------------------------------------

    def test_multiple_rf_windows(self):

        self.add_rf_result(
            prediction="DDoS",
            window_size=10
        )

        self.add_rf_result(
            prediction="BENIGN",
            window_size=10
        )

        self.add_rf_result(
            prediction="DDoS",
            window_size=100
        )

        result = self.finalize(
            is_anomaly=False,
            threat_score=0
        )

        self.assertEqual(
            result["final_prediction"],
            "DDoS"
        )

        self.assertEqual(result["rf_window_count"], 3)

        self.assertEqual(
            result["rf_flagged_window_count"],
            2
        )

        self.assertEqual(
            result["rf_benign_window_count"],
            1
        )

    # -------------------------------------------------
    # TEST 7: RF PREDICTIONS ARE REMOVED AFTER FINALIZATION
    # -------------------------------------------------

    def test_rf_predictions_are_cleaned_up(self):

        self.add_rf_result("DDoS")

        self.finalize()

        self.assertNotIn(
            self.flow_key,
            self.engine.rf_predictions_by_flow
        )

    # -------------------------------------------------
    # TEST 8: MISSING FLOW IDENTIFIER IS REJECTED
    # -------------------------------------------------

    def test_missing_flow_key_is_rejected(self):

        invalid_window = {
            "window_size": 10
        }

        prediction_result = {
            "window_size": 10,
            "prediction": "BENIGN",
            "is_benign": True,
            "model_score": 0.95
        }

        with self.assertRaises(ValueError):

            self.engine.add_rf_prediction(
                invalid_window,
                prediction_result
            )

    # -------------------------------------------------
    # TEST 9: WINDOW SIZE MISMATCH IS REJECTED
    # -------------------------------------------------

    def test_window_size_mismatch_is_rejected(self):

        invalid_window = {
            "flow_key": self.flow_key,
            "window_size": 10
        }

        prediction_result = {
            "window_size": 100,
            "prediction": "BENIGN",
            "is_benign": True,
            "model_score": 0.95
        }

        with self.assertRaises(ValueError):

            self.engine.add_rf_prediction(
                invalid_window,
                prediction_result
            )


if __name__ == "__main__":
    unittest.main()
