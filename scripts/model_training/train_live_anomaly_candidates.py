
"""
Train two candidate Isolation Forest models.

Candidate 1: Original live features.
Candidate 2: Log-transformed live features.

The models are trained using collected traffic assumed
to be normal. This is an initial experiment, not a
measurement of attack detection accuracy.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "live_baseline"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_FILE = DATASET_DIR / "live_features.csv"
LOG_FILE = DATASET_DIR / "live_features_log.csv"

RAW_MODEL_FILE = MODEL_DIR / "live_isolation_forest_raw.pkl"
LOG_MODEL_FILE = MODEL_DIR / "live_isolation_forest_log.pkl"


def train_candidate(dataset_file, model_file, candidate_name):
    """Train, inspect and save one Isolation Forest candidate."""

    # Load the prepared feature matrix.
    df = pd.read_csv(dataset_file)

    # Ensure all model inputs are numerical and valid.
    if df.isnull().values.any():
        raise ValueError(f"Missing values found in {dataset_file}")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid numerical values in {dataset_file}")

    # Create the Isolation Forest model.
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    # Fit the model using our collected baseline traffic.
    model.fit(df)

    # Predict the baseline observations.
    # +1 means normal; -1 means anomalous.
    predictions = model.predict(df)

    # Lower score_samples values indicate more unusual observations.
    anomaly_scores = model.score_samples(df)

    # Decision function values below zero are classified as anomalies.
    decision_scores = model.decision_function(df)

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    anomaly_percentage = (
        anomaly_count / len(df)
    ) * 100

    # Save the model together with its expected input schema.
    model_artifact = {
        "model": model,
        "feature_columns": list(df.columns),
        "candidate_name": candidate_name,
        "contamination": "auto",
        "training_records": len(df),
    }

    joblib.dump(model_artifact, model_file)

    # Display training diagnostics.
    print(f"\n{candidate_name}")
    print("-" * 55)
    print(f"Training records: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Normal predictions: {normal_count}")
    print(f"Anomaly predictions: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

    print("\nAnomaly score distribution:")
    print(
        pd.Series(anomaly_scores)
        .describe()
        .round(4)
        .to_string()
    )

    print("\nDecision score distribution:")
    print(
        pd.Series(decision_scores)
        .describe()
        .round(4)
        .to_string()
    )

    print(f"\nModel saved to: {model_file}")

    return predictions


def main():
    # Create the model directory if it does not exist.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("LIVE ANOMALY DETECTOR TRAINING")
    print("=" * 55)

    # Train using original features.
    raw_predictions = train_candidate(
        RAW_FILE,
        RAW_MODEL_FILE,
        "Isolation Forest - Original Features"
    )

    # Train using log-transformed features.
    log_predictions = train_candidate(
        LOG_FILE,
        LOG_MODEL_FILE,
        "Isolation Forest - Log-Transformed Features"
    )

    # Compare which observations both models identify as anomalous.
    raw_anomalies = set(
        np.where(raw_predictions == -1)[0]
    )

    log_anomalies = set(
        np.where(log_predictions == -1)[0]
    )

    common_anomalies = raw_anomalies.intersection(
        log_anomalies
    )

    print("\nCANDIDATE COMPARISON")
    print("=" * 55)
    print(f"Original model anomalies: {len(raw_anomalies)}")
    print(f"Log model anomalies: {len(log_anomalies)}")
    print(f"Common anomalies: {len(common_anomalies)}")

    print("\nTraining experiment completed.")
    print("These results are not attack detection accuracy.")


if __name__ == "__main__":
    main()
