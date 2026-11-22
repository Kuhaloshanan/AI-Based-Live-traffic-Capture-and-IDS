
"""
Train the selected live Isolation Forest model.

Selected approach:
- Original numerical flow features.
- Protocol excluded from model inputs.
- 200 isolation trees.
- Contamination set to auto.

The model learns patterns from traffic assumed to be normal.
Its anomaly predictions are not confirmed attack classifications.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "live_isolation_forest.pkl"
)

# These are the eight selected live traffic features.
FEATURE_COLUMNS = [
    "duration_seconds",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "min_packet_size",
    "max_packet_size",
    "standard_deviation",
]


def main():

    print("FINAL LIVE ANOMALY DETECTOR TRAINING")
    print("=" * 60)

    # Load the combined baseline traffic.
    df = pd.read_csv(DATASET_FILE)

    # Check that all selected features exist.
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    # Select only the eight numerical features.
    X = df[FEATURE_COLUMNS].copy()

    # Ensure that all model inputs are valid.
    if X.isnull().values.any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Infinite or invalid values detected.")

    # Create the selected Isolation Forest model.
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    # Fit the model using the collected baseline traffic.
    model.fit(X)

    # Inspect how the model treats its training observations.
    predictions = model.predict(X)

    normal_count = int((predictions == 1).sum())
    anomaly_count = int((predictions == -1).sum())

    # Save the model and its input configuration together.
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "Live Isolation Forest",
        "n_estimators": 200,
        "contamination": "auto",
        "random_state": 42,
        "training_records": len(X),
        "protocol_used_as_model_feature": False,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, MODEL_FILE)

    print(f"Training records: {len(X)}")
    print(f"Model input features: {len(FEATURE_COLUMNS)}")
    print(f"Normal training predictions: {normal_count}")
    print(f"Anomalous training predictions: {anomaly_count}")

    print("\nSelected features:")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print(f"\nModel saved to:\n{MODEL_FILE}")

    print("\nLIVE MODEL TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
