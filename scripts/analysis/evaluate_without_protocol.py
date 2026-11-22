
"""
Ablation experiment: train Isolation Forest without protocol features.

This experiment checks whether protocol encoding contributes
to the high anomaly rates observed for rare protocols.

Both collection-session directions are evaluated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "live_baseline"

SESSION_1 = DATASET_DIR / "normal_traffic_20261116_231038.csv"
SESSION_2 = DATASET_DIR / "normal_traffic_20261116_231848.csv"

# Keep only the eight numerical traffic features.
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


def evaluate_direction(train_df, test_df, experiment_name):

    # Prepare training and testing data without protocol columns.
    X_train = train_df[NUMERIC_FEATURES].copy()
    X_test = test_df[NUMERIC_FEATURES].copy()

    # Validate numerical inputs.
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Invalid values in training data.")

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Invalid values in test data.")

    # Train the Isolation Forest using numerical features only.
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    # Predict on the separate collection session.
    predictions = model.predict(X_test)

    # Combine predictions with the original protocol labels.
    results = test_df[["protocol"]].copy()
    results["prediction"] = predictions

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    anomaly_percentage = (
        anomaly_count / len(predictions)
    ) * 100

    print(f"\n{experiment_name}")
    print("=" * 65)

    print(f"Training flows: {len(X_train)}")
    print(f"Testing flows: {len(X_test)}")
    print(f"Normal predictions: {normal_count}")
    print(f"Anomaly predictions: {anomaly_count}")
    print(f"Overall anomaly rate: {anomaly_percentage:.2f}%")

    # Calculate anomaly rates separately for each protocol.
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

    print("\nAnomaly results by protocol:")
    print(summary.round(2).to_string())


def main():

    # Load the two original collection sessions.
    session_1 = pd.read_csv(SESSION_1)
    session_2 = pd.read_csv(SESSION_2)

    print("ISOLATION FOREST PROTOCOL ABLATION STUDY")
    print("=" * 65)

    # Experiment A: Train on session 1, test on session 2.
    evaluate_direction(
        session_1,
        session_2,
        "SESSION 1 -> SESSION 2"
    )

    # Experiment B: Train on session 2, test on session 1.
    evaluate_direction(
        session_2,
        session_1,
        "SESSION 2 -> SESSION 1"
    )


if __name__ == "__main__":
    main()
