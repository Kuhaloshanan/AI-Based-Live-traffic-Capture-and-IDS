
"""
Check mathematical relationships between live flow features.

This helps identify redundant features and verify
that FlowProcessor calculations are consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)


def count_mismatches(actual, expected):
    """Count values that do not match within a small tolerance."""
    matches = np.isclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5
    )

    return int((~matches).sum())


def main():
    df = pd.read_csv(DATASET_FILE)

    print("FEATURE RELATIONSHIP VALIDATION")
    print("=" * 55)

    # Check relationships between packet counts and directions.
    checks = {
        "packet_count = forward_packets + reverse_packets": (
            df["packet_count"],
            df["forward_packets"] + df["reverse_packets"]
        ),
        "total_bytes = forward_bytes + reverse_bytes": (
            df["total_bytes"],
            df["forward_bytes"] + df["reverse_bytes"]
        ),
        "average_packet_size = total_bytes / packet_count": (
            df["average_packet_size"],
            df["total_bytes"] / df["packet_count"]
        ),
        "variance = standard_deviation squared": (
            df["variance"],
            df["standard_deviation"] ** 2
        ),
    }

    for description, (actual, expected) in checks.items():
        mismatches = count_mismatches(actual, expected)

        print(f"\n{description}")
        print(f"  Mismatched rows: {mismatches} / {len(df)}")

    # Rate calculations are only checked for positive durations.
    positive_duration = df["duration_seconds"] > 0

    rate_checks = {
        "packet_rate = packet_count / duration": (
            df.loc[positive_duration, "packet_rate"],
            (
                df.loc[positive_duration, "packet_count"]
                / df.loc[positive_duration, "duration_seconds"]
            )
        ),
        "byte_rate = total_bytes / duration": (
            df.loc[positive_duration, "byte_rate"],
            (
                df.loc[positive_duration, "total_bytes"]
                / df.loc[positive_duration, "duration_seconds"]
            )
        ),
    }

    print("\nRATE RELATIONSHIPS")
    print("-" * 55)

    for description, (actual, expected) in rate_checks.items():
        mismatches = count_mismatches(actual, expected)

        print(f"\n{description}")
        print(
            f"  Mismatched rows: {mismatches} / {len(actual)}"
        )

    # Check basic packet-size consistency.
    invalid_size_rows = (
        (df["min_packet_size"] > df["max_packet_size"])
        | (df["average_packet_size"] < df["min_packet_size"])
        | (df["average_packet_size"] > df["max_packet_size"])
    ).sum()

    print("\nPACKET SIZE VALIDATION")
    print("-" * 55)
    print(f"Invalid packet-size rows: {invalid_size_rows}")

    # Identify highly correlated numerical feature pairs.
    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr().abs()

    upper_triangle = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1
        ).astype(bool)
    )

    high_correlations = (
        upper_triangle
        .stack()
        .sort_values(ascending=False)
    )

    print("\nHIGHLY CORRELATED FEATURE PAIRS")
    print("-" * 55)

    for (feature_a, feature_b), value in high_correlations.items():
        if value >= 0.90:
            print(
                f"{feature_a} <-> {feature_b}: {value:.4f}"
            )


if __name__ == "__main__":
    main()
