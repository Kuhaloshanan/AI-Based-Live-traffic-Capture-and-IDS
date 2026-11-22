
"""
Compare feature skewness before and after log1p transformation.

This helps us decide which traffic features may benefit
from logarithmic transformation before Isolation Forest training.
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

# Only inspect original numerical traffic features.
# Protocol dummy columns are excluded.
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


def main():
    df = pd.read_csv(INPUT_FILE)

    print("LOG TRANSFORMATION ANALYSIS")
    print("=" * 75)

    for feature in NUMERIC_FEATURES:
        values = df[feature]

        # log1p is suitable for non-negative values.
        if (values < 0).any():
            raise ValueError(
                f"{feature} contains negative values."
            )

        transformed = np.log1p(values)

        original_skewness = values.skew()
        transformed_skewness = transformed.skew()

        print(f"\nFeature: {feature}")
        print(f"  Original skewness:    {original_skewness:.4f}")
        print(f"  Log1p skewness:       {transformed_skewness:.4f}")

        print("  Original percentiles:")
        print(
            values.quantile(
                [0.50, 0.90, 0.95, 0.99, 1.00]
            ).round(3).to_dict()
        )

        print("  Log1p percentiles:")
        print(
            transformed.quantile(
                [0.50, 0.90, 0.95, 0.99, 1.00]
            ).round(3).to_dict()
        )


if __name__ == "__main__":
    main()
