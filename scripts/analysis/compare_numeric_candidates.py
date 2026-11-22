
"""
Compare Isolation Forest using numerical features only.

Candidate A: Original numerical features.
Candidate B: Log-transformed numerical features.

Protocol is deliberately excluded from model inputs.
It remains available in the original dataset for analysis.
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


def prepare_features(df, use_log):
    """Prepare numerical features with optional log transformation."""

    features = df[NUMERIC_FEATURES].copy()

    if use_log:
        for column in LOG_FEATURES:
            if (features[column] < 0).any():
                raise ValueError(
                    f"Negative values in {column}"
                )

            features[column] = np.log1p(features[column])

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Invalid feature values detected.")

    return features


def evaluate_candidate(train_df, test_df, use_log, name):
    """Train on one session and evaluate on another."""

    X_train = prepare_features(train_df, use_log)
    X_test = prepare_features(test_df, use_log)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    predictions = model.predict(X_test)

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    anomaly_rate = anomaly_count / len(X_test) * 100

    print(f"\n{name}")
    print("-" * 60)
    print(f"Training flows: {len(X_train)}")
    print(f"Testing flows: {len(X_test)}")
    print(f"Normal flows: {normal_count}")
    print(f"Anomalous flows: {anomaly_count}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")

    return predictions


def run_experiment(train_df, test_df, direction):

    print(f"\n{'=' * 65}")
    print(direction)
    print("=" * 65)

    raw_predictions = evaluate_candidate(
        train_df,
        test_df,
        use_log=False,
        name="Original numerical features"
    )

    log_predictions = evaluate_candidate(
        train_df,
        test_df,
        use_log=True,
        name="Log-transformed numerical features"
    )

    # Compare the observations flagged by both candidates.
    raw_anomalies = set(
        np.where(raw_predictions == -1)[0]
    )

    log_anomalies = set(
        np.where(log_predictions == -1)[0]
    )

    print("\nPrediction overlap")
    print(f"Original anomalies: {len(raw_anomalies)}")
    print(f"Log anomalies: {len(log_anomalies)}")
    print(
        "Common anomalies:",
        len(raw_anomalies.intersection(log_anomalies))
    )


def main():

    session_1 = pd.read_csv(SESSION_1)
    session_2 = pd.read_csv(SESSION_2)

    print("NUMERICAL FEATURE PREPROCESSING COMPARISON")
    print("Protocol features excluded from both candidates.")

    # Train on session 1 and test on session 2.
    run_experiment(
        session_1,
        session_2,
        "SESSION 1 -> SESSION 2"
    )

    # Train on session 2 and test on session 1.
    run_experiment(
        session_2,
        session_1,
        "SESSION 2 -> SESSION 1"
    )


if __name__ == "__main__":
    main()
