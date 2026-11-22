
"""
Analyze anomaly flag rates by network protocol.

Tests both Isolation Forest candidates across
the two collection-session directions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "live_baseline"

SESSION_1 = DATASET_DIR / "normal_traffic_20261116_231038.csv"
SESSION_2 = DATASET_DIR / "normal_traffic_20261116_231848.csv"

NUMERIC_FEATURES = [
    "duration_seconds",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "min_packet_size",
    "max_packet_size",
    "standard_deviation",
]

LOG_FEATURES = [
    "duration_seconds",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "min_packet_size",
]

PROTOCOLS = ["TCP", "UDP", "OTHER", "ICMPv6"]

FEATURE_COLUMNS = NUMERIC_FEATURES + [
    f"protocol_{protocol}"
    for protocol in PROTOCOLS
]


def prepare_features(df, use_log):
    """Prepare features using the established project schema."""

    features = df[NUMERIC_FEATURES + ["protocol"]].copy()

    features["protocol"] = pd.Categorical(
        features["protocol"],
        categories=PROTOCOLS
    )

    features = pd.get_dummies(
        features,
        columns=["protocol"],
        prefix="protocol",
        dtype=int
    )

    features = features.reindex(
        columns=FEATURE_COLUMNS
    )

    if use_log:
        for column in LOG_FEATURES:
            features[column] = np.log1p(
                features[column]
            )

    return features


def analyze_direction(train_df, test_df, direction):
    """Train each candidate and summarize test anomalies."""

    for use_log, candidate in [
        (False, "Original"),
        (True, "Log-transformed")
    ]:

        train_features = prepare_features(
            train_df,
            use_log
        )

        test_features = prepare_features(
            test_df,
            use_log
        )

        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )

        model.fit(train_features)

        predictions = model.predict(test_features)

        # Attach predictions to the original protocol labels.
        results = test_df[["protocol"]].copy()
        results["prediction"] = predictions

        # Calculate protocol-specific anomaly counts.
        summary = (
            results.groupby("protocol", observed=False)
            .agg(
                total_flows=("prediction", "size"),
                anomalous_flows=(
                    "prediction",
                    lambda values: (values == -1).sum()
                )
            )
        )

        summary["anomaly_percentage"] = (
            summary["anomalous_flows"]
            / summary["total_flows"]
            * 100
        )

        print(f"\n{direction} | {candidate} features")
        print("-" * 65)
        print(
            summary.round(2).to_string()
        )


def main():
    session_1 = pd.read_csv(SESSION_1)
    session_2 = pd.read_csv(SESSION_2)

    print("PROTOCOL-LEVEL ANOMALY ANALYSIS")
    print("=" * 65)

    analyze_direction(
        session_1,
        session_2,
        "SESSION 1 -> SESSION 2"
    )

    analyze_direction(
        session_2,
        session_1,
        "SESSION 2 -> SESSION 1"
    )


if __name__ == "__main__":
    main()
