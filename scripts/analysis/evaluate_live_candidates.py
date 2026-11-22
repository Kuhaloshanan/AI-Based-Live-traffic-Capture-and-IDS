
"""
Evaluate Isolation Forest candidates across separate
network traffic collection sessions.

Experiment A:
    Train on session 1, test on session 2.

Experiment B:
    Train on session 2, test on session 1.

Both original and log-transformed features are evaluated.

The test sessions are assumed normal, not independently
verified as benign. Therefore, anomaly percentages are
not confirmed false-positive rates.
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

PROTOCOL_CATEGORIES = [
    "TCP",
    "UDP",
    "OTHER",
    "ICMPv6",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + [
    "protocol_TCP",
    "protocol_UDP",
    "protocol_OTHER",
    "protocol_ICMPv6",
]


def prepare_features(df, use_log):
    """Apply the same feature preparation to each session."""

    features = df[NUMERIC_FEATURES + ["protocol"]].copy()

    # Use a fixed category order across both sessions.
    features["protocol"] = pd.Categorical(
        features["protocol"],
        categories=PROTOCOL_CATEGORIES
    )

    # Convert protocol into numerical binary columns.
    features = pd.get_dummies(
        features,
        columns=["protocol"],
        prefix="protocol",
        dtype=int
    )

    # Guarantee identical feature names and order.
    features = features.reindex(
        columns=FEATURE_COLUMNS
    )

    # Apply the selected logarithmic transformation.
    if use_log:
        for column in LOG_FEATURES:
            if (features[column] < 0).any():
                raise ValueError(
                    f"Negative values found in {column}"
                )

            features[column] = np.log1p(
                features[column]
            )

    # Final numerical validation.
    if not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid values found in prepared features."
        )

    return features


def evaluate_candidate(
    train_features,
    test_features,
    candidate_name,
    experiment_name
):
    """Train on one session and evaluate on another."""

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_features)

    # Evaluate on training and unseen test observations.
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    test_scores = model.score_samples(test_features)

    train_anomalies = int(
        (train_predictions == -1).sum()
    )

    test_anomalies = int(
        (test_predictions == -1).sum()
    )

    test_percentage = (
        test_anomalies / len(test_features)
    ) * 100

    print(f"\n{experiment_name} | {candidate_name}")
    print("-" * 65)

    print(f"Training records: {len(train_features)}")
    print(f"Test records: {len(test_features)}")

    print(f"Training anomalies: {train_anomalies}")
    print(f"Test anomalies: {test_anomalies}")

    print(
        f"Test anomaly flag rate: {test_percentage:.2f}%"
    )

    print(
        f"Test score minimum: {test_scores.min():.4f}"
    )

    print(
        f"Test score median: {np.median(test_scores):.4f}"
    )

    print(
        f"Test score maximum: {test_scores.max():.4f}"
    )


def run_experiment(
    train_df,
    test_df,
    experiment_name
):
    """Compare original and transformed candidates."""

    raw_train = prepare_features(
        train_df,
        use_log=False
    )

    raw_test = prepare_features(
        test_df,
        use_log=False
    )

    log_train = prepare_features(
        train_df,
        use_log=True
    )

    log_test = prepare_features(
        test_df,
        use_log=True
    )

    evaluate_candidate(
        raw_train,
        raw_test,
        "Original Features",
        experiment_name
    )

    evaluate_candidate(
        log_train,
        log_test,
        "Log-Transformed Features",
        experiment_name
    )


def main():
    print("CROSS-SESSION ANOMALY DETECTOR EVALUATION")
    print("=" * 70)

    # Read the two independent collection sessions.
    session_1 = pd.read_csv(SESSION_1)
    session_2 = pd.read_csv(SESSION_2)

    print(f"Session 1 records: {len(session_1)}")
    print(f"Session 2 records: {len(session_2)}")

    # Experiment A: Train on the smaller session.
    run_experiment(
        session_1,
        session_2,
        "EXPERIMENT A: SESSION 1 -> SESSION 2"
    )

    # Experiment B: Reverse the training and test sessions.
    run_experiment(
        session_2,
        session_1,
        "EXPERIMENT B: SESSION 2 -> SESSION 1"
    )

    print("\nCROSS-SESSION EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
