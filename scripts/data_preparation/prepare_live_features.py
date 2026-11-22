
"""
Prepare numerical features for the live Isolation Forest.

We remove mathematically derived features and encode
the protocol column using one-hot encoding.

The original combined dataset remains unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "live_features.csv"
)

# Numerical features selected for the initial experiment.
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

# Keep a fixed protocol order for consistent encoding.
PROTOCOL_CATEGORIES = [
    "TCP",
    "UDP",
    "OTHER",
    "ICMPv6",
]


def main():
    print("LIVE FEATURE PREPARATION")
    print("-" * 50)

    # Load the original combined dataset.
    df = pd.read_csv(INPUT_FILE)

    # Ensure all required columns are available.
    required_columns = NUMERIC_FEATURES + ["protocol"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Reject unexpected protocols rather than silently
    # encoding them as all-zero values.
    unknown_protocols = (
        set(df["protocol"].dropna().unique())
        - set(PROTOCOL_CATEGORIES)
    )

    if unknown_protocols:
        raise ValueError(
            f"Unexpected protocols found: {unknown_protocols}"
        )

    # Select only the features chosen for the experiment.
    features = df[NUMERIC_FEATURES + ["protocol"]].copy()

    # Convert protocol values into a fixed categorical type.
    features["protocol"] = pd.Categorical(
        features["protocol"],
        categories=PROTOCOL_CATEGORIES
    )

    # Convert protocol into four binary columns.
    features = pd.get_dummies(
        features,
        columns=["protocol"],
        prefix="protocol",
        dtype=int
    )

    # Check for missing and infinite numerical values.
    missing_values = int(features.isnull().sum().sum())

    infinite_values = int(
        np.isinf(features.to_numpy(dtype=float)).sum()
    )

    if missing_values > 0 or infinite_values > 0:
        raise ValueError(
            "Prepared features contain missing or infinite values."
        )

    # Save the prepared feature matrix.
    features.to_csv(OUTPUT_FILE, index=False)

    print(f"Original flow records: {len(df)}")
    print(f"Selected numerical features: {len(NUMERIC_FEATURES)}")
    print("Protocol encoding: 4 columns")
    print(f"Prepared feature count: {len(features.columns)}")
    print(f"Missing values: {missing_values}")
    print(f"Infinite values: {infinite_values}")

    print("\nPrepared feature columns:")
    for column in features.columns:
        print(f"  - {column}")

    print(f"\nSaved prepared dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
