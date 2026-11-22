
"""
Apply logarithmic transformation to highly skewed features.

The original live_features.csv is never modified.
A separate transformed dataset is created for comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "live_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "live_features_log.csv"
)

# Features selected for logarithmic transformation.
LOG_FEATURES = [
    "duration_seconds",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "min_packet_size",
]


def main():
    # Load the original prepared dataset.
    df = pd.read_csv(INPUT_FILE)

    # Work on a copy so the original data remains unchanged.
    transformed_df = df.copy()

    # Apply log1p only to the selected non-negative features.
    for feature in LOG_FEATURES:

        if (transformed_df[feature] < 0).any():
            raise ValueError(
                f"Negative values found in {feature}."
            )

        transformed_df[feature] = np.log1p(
            transformed_df[feature]
        )

    # Verify the transformed dataset.
    missing_values = int(
        transformed_df.isnull().sum().sum()
    )

    infinite_values = int(
        np.isinf(
            transformed_df.to_numpy(dtype=float)
        ).sum()
    )

    if missing_values > 0 or infinite_values > 0:
        raise ValueError(
            "Transformed dataset contains invalid values."
        )

    # Save as a separate dataset.
    transformed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("LOG TRANSFORMATION COMPLETED")
    print("-" * 50)

    print(f"Total records: {len(transformed_df)}")
    print(f"Total features: {len(transformed_df.columns)}")
    print(f"Transformed features: {len(LOG_FEATURES)}")
    print(f"Missing values: {missing_values}")
    print(f"Infinite values: {infinite_values}")

    print("\nFeatures transformed:")
    for feature in LOG_FEATURES:
        print(f"  - {feature}")

    print(f"\nSaved transformed dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
