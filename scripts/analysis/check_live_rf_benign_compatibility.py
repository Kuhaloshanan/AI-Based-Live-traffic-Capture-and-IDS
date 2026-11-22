
"""Compare live windows specifically against BENIGN training records."""

import pandas as pd
import numpy as np

TRAIN_PATH = "dataset/MERGED_CSV/Merged01.csv"

LIVE_PATH = (
    "dataset/live_rf_validation/"
    "live_rf_windows_20261122_002930.csv"
)

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
]


def main():

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["Label"] + FEATURES
    )

    live = pd.read_csv(LIVE_PATH)

    train["Number"] = pd.to_numeric(
        train["Number"], errors="coerce"
    )

    train["Label"] = train["Label"].astype(str).str.strip()

    print("\n========== LABEL DISTRIBUTION ==========")

    for size in (10, 100):

        training_group = train[
            train["Number"] == size
        ]

        live_group = live[
            live["window_size"] == size
        ]

        benign_group = training_group[
            training_group["Label"].str.upper() == "BENIGN"
        ]

        print(f"\nWINDOW SIZE: {size}")
        print("All training records:", len(training_group))
        print("BENIGN training records:", len(benign_group))
        print("Live records:", len(live_group))

        print("\nTraining class distribution:")
        print(training_group["Label"].value_counts().head(12))

        if benign_group.empty:
            print("\nNo BENIGN training records for this window size.")
            continue

        print("\nBENIGN training vs live median:")

        for feature in FEATURES:

            train_values = pd.to_numeric(
                benign_group[feature], errors="coerce"
            )

            live_values = pd.to_numeric(
                live_group[feature], errors="coerce"
            )

            print(
                f"{feature:20s} "
                f"BENIGN={train_values.median():12.4f} "
                f"LIVE={live_values.median():12.4f}"
            )

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
