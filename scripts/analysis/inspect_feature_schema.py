
"""
Feature Schema Investigation

Investigate mathematical relationships between features
in the CICIoT2023 CSV file.

This helps us understand how the dataset represents
packet-window statistics before implementing live extraction.
"""

import pandas as pd


# Dataset location
DATA_PATH = "dataset/MERGED_CSV/Merged01.csv"


# Read only the columns required for this investigation.
columns = [
    "Number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "ack_flag_number",
    "syn_flag_number",
    "fin_flag_number",
    "rst_flag_number",
    "Tot sum",
    "AVG",
    "Tot size",
    "Std",
    "Variance"
]


def check_relationship(df, feature, calculated_value):
    """
    Compare an existing feature with a calculated value.
    """

    difference = (
        df[feature] - calculated_value
    ).abs()

    # Count records matching within a small tolerance.
    matches = difference < 0.000001

    print(f"\nFeature: {feature}")
    print(f"Matching records: {matches.sum():,}")
    print(f"Total records: {len(df):,}")
    print(f"Match percentage: {matches.mean() * 100:.2f}%")
    print(f"Maximum difference: {difference.max():.8f}")


def main():

    print("Loading selected dataset columns...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns
    )

    # Exclude records where Number is zero to avoid division by zero.
    df = df[df["Number"] > 0].copy()

    print("Records inspected:", len(df))

    # Investigate TCP flag proportions.
    check_relationship(
        df,
        "ack_flag_number",
        df["ack_count"] / df["Number"]
    )

    check_relationship(
        df,
        "syn_flag_number",
        df["syn_count"] / df["Number"]
    )

    check_relationship(
        df,
        "fin_flag_number",
        df["fin_count"] / df["Number"]
    )

    check_relationship(
        df,
        "rst_flag_number",
        df["rst_count"] / df["Number"]
    )

    # Investigate packet-size calculations.
    check_relationship(
        df,
        "AVG",
        df["Tot sum"] / df["Number"]
    )

    # Investigate variance and standard deviation.
    check_relationship(
        df,
        "Variance",
        df["Std"] ** 2
    )

    # Investigate whether Tot size equals AVG.
    check_relationship(
        df,
        "Tot size",
        df["AVG"]
    )


if __name__ == "__main__":
    main()
