
"""Train a hybrid Random Forest for live IPv4 traffic classification.

Training sources:
1. CICIoT2023 labelled training data.
2. Newly collected assumed-BENIGN live traffic.

An earlier live capture is kept separate for validation.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.live_rf_window_processor import RF_FEATURE_COLUMNS


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

PROCESSED_DIR = Path("dataset/processed")
LIVE_DIR = Path("dataset/live_rf_validation")
MODEL_DIR = Path("models")

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"

X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

# New 10-minute capture: used for training.
LIVE_TRAIN_PATH = (
    LIVE_DIR / "live_rf_windows_20261122_013446.csv"
)

# Earlier two-minute capture: kept separate for validation.
LIVE_VALIDATION_PATH = (
    LIVE_DIR / "live_rf_windows_20261122_012041.csv"
)

MODEL_PATH = MODEL_DIR / "random_forest_live_hybrid.pkl"
REPORT_PATH = LIVE_DIR / "live_rf_hybrid_evaluation.txt"


# --------------------------------------------------
# DATA LOADING FUNCTIONS
# --------------------------------------------------

def load_features(path):
    """Load and validate the 39 RF input features."""

    dataframe = pd.read_csv(path)

    # Remove accidental CSV index columns if present.
    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str).str.startswith("Unnamed")
    ]

    missing = set(RF_FEATURE_COLUMNS) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path} is missing features: {sorted(missing)}"
        )

    # Preserve the exact feature order expected by the model.
    dataframe = dataframe[RF_FEATURE_COLUMNS].copy()

    for column in RF_FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column], errors="coerce"
        )

    values = dataframe.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing or infinite values found in {path}"
        )

    return dataframe


def load_labels(path):
    """Load labels from a one-column CSV."""

    dataframe = pd.read_csv(path)

    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str).str.startswith("Unnamed")
    ]

    if "Label" in dataframe.columns:
        labels = dataframe["Label"]
    elif dataframe.shape[1] == 1:
        labels = dataframe.iloc[:, 0]
    else:
        raise ValueError(
            f"Cannot identify label column in {path}"
        )

    return labels.astype(str).str.strip().reset_index(drop=True)


# --------------------------------------------------
# MAIN TRAINING PROCESS
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== HYBRID RANDOM FOREST TRAINING ==========")

    # Load original CICIoT2023 training data.
    print("\nLoading CICIoT2023 training data...")

    X_train = load_features(X_TRAIN_PATH)
    y_train = load_labels(Y_TRAIN_PATH)

    if len(X_train) != len(y_train):
        raise ValueError("Training feature and label counts differ.")

    print("CICIoT training rows:", len(X_train))

    # Load the new live traffic.
    print("\nLoading assumed-BENIGN live training data...")

    live_train = pd.read_csv(LIVE_TRAIN_PATH)

    if "window_size" not in live_train.columns:
        raise ValueError("Live training file has no window_size column.")

    if not (live_train["IPv"] == 1).all():
        raise ValueError("Live training data contains non-IPv4 windows.")

    X_live = load_features(LIVE_TRAIN_PATH)
    y_live = pd.Series(
        ["BENIGN"] * len(X_live),
        name="Label"
    )

    print("Live BENIGN training rows:", len(X_live))
    print(
        "Live window distribution:\n",
        live_train["window_size"].value_counts().sort_index()
    )

    # Combine the two training sources.
    X_combined = pd.concat(
        [X_train, X_live],
        ignore_index=True
    )

    y_combined = pd.concat(
        [y_train, y_live],
        ignore_index=True
    )

    print("\nCombined training rows:", len(X_combined))

    print("\nTraining class distribution:")
    print(y_combined.value_counts())

    # Configure the Random Forest.
    print("\nInitializing Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    # Train the new hybrid classifier.
    print("\nTraining started. Please wait...")

    model.fit(X_combined, y_combined)

    print("Training completed.")

    # --------------------------------------------------
    # EVALUATION 1: ORIGINAL CICIoT TEST DATA
    # --------------------------------------------------

    print("\n========== CICIoT TEST EVALUATION ==========")

    X_test = load_features(X_TEST_PATH)
    y_test = load_labels(Y_TEST_PATH)

    if len(X_test) != len(y_test):
        raise ValueError("Test feature and label counts differ.")

    predictions = model.predict(X_test)

    cic_accuracy = accuracy_score(y_test, predictions)
    cic_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    cic_report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("Accuracy:", round(cic_accuracy, 4))
    print("Macro F1:", round(cic_macro_f1, 4))
    print("\nClassification report:")
    print(cic_report)

    # --------------------------------------------------
    # EVALUATION 2: INDEPENDENT LIVE BENIGN CAPTURE
    # --------------------------------------------------

    print("\n========== LIVE TRAFFIC VALIDATION ==========")

    live_validation = pd.read_csv(LIVE_VALIDATION_PATH)

    if "window_size" not in live_validation.columns:
        raise ValueError("Live validation file has no window_size column.")

    X_live_test = load_features(LIVE_VALIDATION_PATH)

    if not (X_live_test["IPv"] == 1).all():
        raise ValueError("Validation file contains non-IPv4 windows.")

    live_predictions = model.predict(X_live_test)

    live_prediction_counts = pd.Series(
        live_predictions
    ).value_counts()

    benign_predictions = int(
        np.sum(live_predictions == "BENIGN")
    )

    assumed_benign_flagged = (
        len(live_predictions) - benign_predictions
    )

    print("Independent live windows:", len(X_live_test))
    print("Predicted BENIGN:", benign_predictions)
    print("Predicted non-BENIGN:", assumed_benign_flagged)

    if len(live_predictions) > 0:
        print(
            "Assumed-BENIGN flagged fraction:",
            round(assumed_benign_flagged / len(live_predictions), 4)
        )

    print("\nLive predicted categories:")
    print(live_prediction_counts)

    print("\nLive predictions by window size:")

    live_results = pd.DataFrame({
        "window_size": live_validation["window_size"],
        "prediction": live_predictions,
    })

    print(
        pd.crosstab(
            live_results["window_size"],
            live_results["prediction"]
        )
    )

    # --------------------------------------------------
    # SAVE MODEL AND EVALUATION REPORT
    # --------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    report_text = (
        "HYBRID RANDOM FOREST EVALUATION\n"
        "================================\n\n"
        f"Combined training rows: {len(X_combined)}\n"
        f"CICIoT test accuracy: {cic_accuracy:.4f}\n"
        f"CICIoT test macro F1: {cic_macro_f1:.4f}\n\n"
        "CICIoT classification report:\n"
        f"{cic_report}\n\n"
        "Independent live assumed-BENIGN validation:\n"
        f"Total windows: {len(X_live_test)}\n"
        f"Predicted BENIGN: {benign_predictions}\n"
        f"Predicted non-BENIGN: {assumed_benign_flagged}\n\n"
        "NOTE: Live validation labels are assumed BENIGN, "
        "not independently verified ground truth.\n"
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8"
    )

    print("\n========== TRAINING SUMMARY ==========")
    print("Saved model:", MODEL_PATH)
    print("Saved report:", REPORT_PATH)
    print("Original models were not modified.")


if __name__ == "__main__":
    main()
