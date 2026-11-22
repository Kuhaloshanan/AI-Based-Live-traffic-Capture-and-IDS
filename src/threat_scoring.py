
"""
Explainable Threat Scoring Engine.

Uses:
1. Isolation Forest anomaly classification.
2. Empirical normal-traffic baseline percentiles.
3. Transparent application-level scoring rules.

The threat score represents priority for investigation,
NOT the probability that a cyberattack occurred.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASELINE_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)

REQUIRED_COLUMNS = [
    "duration_seconds",
    "packet_rate",
    "byte_rate",
]


class ThreatScorer:

    def __init__(self, baseline_path=DEFAULT_BASELINE_FILE):

        self.baseline_path = Path(baseline_path)

        if not self.baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline file not found: {self.baseline_path}"
            )

        # Load baseline only once.
        baseline = pd.read_csv(self.baseline_path)

        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in baseline.columns
        ]

        if missing:
            raise ValueError(
                f"Baseline is missing columns: {missing}"
            )

        baseline = baseline[REQUIRED_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(baseline) < 20:
            raise ValueError(
                "Insufficient baseline records for scoring."
            )

        # Cache baseline values and calculate thresholds.
        self.baseline_values = {}
        self.thresholds = {}

        for metric in REQUIRED_COLUMNS:

            values = baseline[metric].to_numpy(dtype=float)

            if np.any(values < 0):
                raise ValueError(
                    f"Baseline contains negative values: {metric}"
                )

            self.baseline_values[metric] = values

            self.thresholds[metric] = {
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }

        self.training_records = len(baseline)

        print(
            f"Threat Scorer initialized with "
            f"{self.training_records} baseline records."
        )

    # -----------------------------------------------------
    # VALIDATE FLOW FEATURES
    # -----------------------------------------------------

    @staticmethod
    def _get_value(flow, key):

        if key not in flow:
            raise ValueError(
                f"Missing flow feature: {key}"
            )

        value = float(flow[key])

        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid flow feature: {key}"
            )

        return value

    # -----------------------------------------------------
    # DETERMINE THREAT LEVEL
    # -----------------------------------------------------

    @staticmethod
    def _get_threat_level(score):

        if score < 25:
            return "LOW"

        if score < 50:
            return "MEDIUM"

        if score < 75:
            return "HIGH"

        return "CRITICAL"

    # -----------------------------------------------------
    # CALCULATE THREAT PRIORITY SCORE
    # -----------------------------------------------------

    def score_flow(self, flow, detection_result):

        if "is_anomaly" not in detection_result:
            raise ValueError(
                "Detection result is missing is_anomaly."
            )

        # Normal traffic receives no anomaly priority.
        if not detection_result["is_anomaly"]:

            return {
                "threat_score": 0,
                "threat_level": "LOW",
                "score_reasons": [
                    "Isolation Forest classified this flow as NORMAL."
                ],
                "baseline_percentiles": {},
            }

        score = 20

        reasons = [
            "Isolation Forest classified this flow as ANOMALY (+20)."
        ]

        values = {
            "duration_seconds": self._get_value(
                flow, "duration_seconds"
            ),
            "packet_rate": self._get_value(
                flow, "packet_rate"
            ),
            "byte_rate": self._get_value(
                flow, "byte_rate"
            ),
        }

        extreme_metrics = []
        baseline_percentiles = {}

        for metric, value in values.items():

            thresholds = self.thresholds[metric]

            # Calculate the empirical percentile rank.
            baseline_values = self.baseline_values[metric]

            percentile = (
                np.mean(baseline_values <= value) * 100
            )

            baseline_percentiles[metric] = round(
                float(percentile), 2
            )

            # High traffic duration or volume.
            if value > thresholds["p99"]:

                score += 20
                extreme_metrics.append(metric)

                reasons.append(
                    f"{metric} exceeds baseline P99 (+20)."
                )

            elif value > thresholds["p95"]:

                score += 10
                extreme_metrics.append(metric)

                reasons.append(
                    f"{metric} exceeds baseline P95 (+10)."
                )

            # Unusually low packet or byte rate.
            elif (
                metric in ("packet_rate", "byte_rate")
                and value < thresholds["p05"]
            ):

                score += 10
                extreme_metrics.append(metric)

                reasons.append(
                    f"{metric} is below baseline P05 (+10)."
                )

        # Additional priority for multiple unusual metrics.
        if len(extreme_metrics) >= 2:

            score += 10

            reasons.append(
                "Two or more unusual characteristics (+10)."
            )

        if len(extreme_metrics) == 3:

            score += 10

            reasons.append(
                "All three characteristics are unusual (+10)."
            )

        score = min(score, 100)

        return {
            "threat_score": score,
            "threat_level": self._get_threat_level(score),
            "score_reasons": reasons,
            "baseline_percentiles": baseline_percentiles,
        }
