
"""Audit combined live windows against CICIoT2023 BENIGN records."""

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_PATH = Path("dataset/MERGED_CSV/Merged01.csv")
LIVE_PATH = Path(
    "dataset/live_rf_validation/live_rf_windows_combined.csv"
)

# Features selected to compare important traffic characteristics.
FEATURES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "TCP",
    "UDP",
    "HTTPS",
    "IPv",
    "Tot sum",
    "AVG",
    "Max",
    "Std",
    "IAT",
    "Number",
    "Variance",
]


def main():
    print("Loading CICIoT2023 training data...")
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["Label"] + FEATURES
    )

    print("Loading combined live captures...")
    live = pd.read_csv(LIVE_PATH)

    # Convert feature columns to numeric values.
    for column in FEATURES:
        train[column] = pd.to_numeric(
            train[column], errors="coerce"
        )
        live[column] = pd.to_numeric(
            live[column], errors="coerce"
        )

    train["Number"] = train["Number"].round()
    live["Number"] = live["Number"].round()
    live["window_size"] = pd.to_numeric(
        live["window_size"], errors="coerce"
    )

    # Check data quality before comparing distributions.
    print("\n========== DATA QUALITY ==========")
    print("Training rows:", len(train))
    print("Live rows:", len(live))
    print("Training missing values:", train[FEATURES].isna().sum().sum())
    print("Live missing values:", live[FEATURES].isna().sum().sum())

    train_infinite = np.isinf(train[FEATURES].to_numpy()).sum()
    live_infinite = np.isinf(live[FEATURES].to_numpy()).sum()

    print("Training infinite values:", train_infinite)
    print("Live infinite values:", live_infinite)

    # Verify that Number agrees with the packet-window size.
    mismatch_count = (
        live["Number"] != live["window_size"]
    ).sum()

    print("Live Number/window_size mismatches:", mismatch_count)

    # Compare the training labels and live windows separately.
    for size in (10, 100):
        training_group = train[
            train["Number"] == size
        ]

        benign_group = training_group[
            training_group["Label"]
            .astype(str).str.strip().str.upper() == "BENIGN"
        ]

        live_group = live[
            live["window_size"] == size
        ]

        print(f"\n========== {size}-PACKET WINDOWS ==========")
        print("Training records:", len(training_group))
        print("Training BENIGN records:", len(benign_group))
        print("Live assumed-BENIGN records:", len(live_group))

        print("\nTraining class distribution:")
        print(training_group["Label"].value_counts().head(10))

        if benign_group.empty or live_group.empty:
            print("Cannot compare: one group is empty.")
            continue

        print("\nFeature distribution comparison")
        print("(P10 = lower range, P50 = median, P90 = upper range)")
        print(
            f"{'Feature':20s}"
            f"{'BENIGN P50':>14s}"
            f"{'LIVE P50':>14s}"
            f"{'BENIGN P90':>14s}"
            f"{'LIVE P90':>14s}"
        )

        for feature in FEATURES:
            benign_values = benign_group[feature].dropna()
            live_values = live_group[feature].dropna()

            print(
                f"{feature:20s}"
                f"{benign_values.median():14.4f}"
                f"{live_values.median():14.4f}"
                f"{benign_values.quantile(0.90):14.4f}"
                f"{live_values.quantile(0.90):14.4f}"
            )

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
