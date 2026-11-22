
"""
Inspect anomalous flows identified by both Isolation Forest models.

compare:
1. Flows flagged by both models.
2. Flows flagged only by the log-transformed model.

Lower score_samples values indicate more unusual observations.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "live_baseline"
MODEL_DIR = PROJECT_ROOT / "models"

BASELINE_FILE = DATASET_DIR / "normal_traffic_combined.csv"
RAW_FEATURE_FILE = DATASET_DIR / "live_features.csv"
LOG_FEATURE_FILE = DATASET_DIR / "live_features_log.csv"

RAW_MODEL_FILE = MODEL_DIR / "live_isolation_forest_raw.pkl"
LOG_MODEL_FILE = MODEL_DIR / "live_isolation_forest_log.pkl"

TOP_N = 10


def main():
    # Load the original flow information.
    baseline_df = pd.read_csv(BASELINE_FILE)

    # Load both prepared feature matrices.
    raw_features = pd.read_csv(RAW_FEATURE_FILE)
    log_features = pd.read_csv(LOG_FEATURE_FILE)

    # Load both trained model artifacts.
    raw_artifact = joblib.load(RAW_MODEL_FILE)
    log_artifact = joblib.load(LOG_MODEL_FILE)

    raw_model = raw_artifact["model"]
    log_model = log_artifact["model"]

    # Check that all files refer to the same observations.
    if not (
        len(baseline_df)
        == len(raw_features)
        == len(log_features)
    ):
        raise ValueError("Dataset row counts do not match.")

    if list(raw_features.columns) != raw_artifact["feature_columns"]:
        raise ValueError("Original feature schema mismatch.")

    if list(log_features.columns) != log_artifact["feature_columns"]:
        raise ValueError("Log feature schema mismatch.")

    # Obtain predictions and anomaly scores.
    raw_predictions = raw_model.predict(raw_features)
    log_predictions = log_model.predict(log_features)

    raw_scores = raw_model.score_samples(raw_features)
    log_scores = log_model.score_samples(log_features)

    # Combine original traffic information and model results.
    results = baseline_df.copy()

    results.insert(0, "flow_id", range(len(results)))

    results["raw_prediction"] = raw_predictions
    results["log_prediction"] = log_predictions

    results["raw_score"] = raw_scores
    results["log_score"] = log_scores

    # Identify flows flagged only by the log model.
    log_only = results[
        (results["raw_prediction"] == 1)
        & (results["log_prediction"] == -1)
    ].copy()

    # Identify flows flagged by both models.
    shared = results[
        (results["raw_prediction"] == -1)
        & (results["log_prediction"] == -1)
    ].copy()

    # Lower scores indicate more unusual observations.
    log_only = log_only.sort_values("log_score")
    shared = shared.sort_values("log_score")

    display_columns = [
        "flow_id",
        "protocol",
        "duration_seconds",
        "packet_count",
        "total_bytes",
        "forward_packets",
        "reverse_packets",
        "forward_bytes",
        "reverse_bytes",
        "min_packet_size",
        "max_packet_size",
        "packet_rate",
        "byte_rate",
        "raw_score",
        "log_score",
    ]

    print("ANOMALY INSPECTION")
    print("=" * 70)

    print(f"Log-only anomalies: {len(log_only)}")
    print(f"Shared anomalies: {len(shared)}")

    print("\nTOP LOG-ONLY ANOMALIES")
    print("-" * 70)

    if not log_only.empty:
        print(
            log_only[display_columns]
            .head(TOP_N)
            .round(4)
            .to_string(index=False)
        )

    print("\nTOP SHARED ANOMALIES")
    print("-" * 70)

    if not shared.empty:
        print(
            shared[display_columns]
            .head(TOP_N)
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
