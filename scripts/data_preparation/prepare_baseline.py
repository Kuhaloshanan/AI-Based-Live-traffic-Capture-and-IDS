
"""
Prepare the live network baseline dataset.

This script:
1. Finds timestamped baseline CSV files.
2. Checks that they have the expected feature columns.
3. Combines the collection sessions.
4. Validates missing and infinite values.
5. Saves one combined CSV file.

We deliberately preserve duplicate rows and zero-duration flows.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# Project directory and baseline data directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_DIR = PROJECT_ROOT / "dataset" / "live_baseline"

# Expected 15 features produced by FlowProcessor.
EXPECTED_COLUMNS = [
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
    "average_packet_size",
    "standard_deviation",
    "variance",
    "packet_rate",
    "byte_rate",
]

# Only match original timestamped collection files.
# This prevents the combined output from being included next time.
FILE_PATTERN = re.compile(
    r"^normal_traffic_\d{8}_\d{6}\.csv$"
)

OUTPUT_FILE = BASELINE_DIR / "normal_traffic_combined.csv"


def main():
    # Find the original baseline collection files.
    csv_files = sorted(
        file
        for file in BASELINE_DIR.glob("*.csv")
        if FILE_PATTERN.match(file.name)
    )

    if not csv_files:
        raise FileNotFoundError(
            "No timestamped baseline CSV files were found."
        )

    dataframes = []

    print("BASELINE DATA PREPARATION")
    print("-" * 40)

    # Read and validate each collection session.
    for file in csv_files:
        print(f"Reading: {file.name}")

        df = pd.read_csv(file)

        # Ensure every session has the same feature schema.
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {file.name}.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {list(df.columns)}"
            )

        print(f"  Rows: {len(df)}")
        dataframes.append(df)

    # Combine all sessions without deleting any records.
    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Check data quality.
    missing_values = int(combined_df.isnull().sum().sum())

    numeric_df = combined_df.select_dtypes(
        include=[np.number]
    )

    infinite_values = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    duplicate_rows = int(
        combined_df.duplicated().sum()
    )

    zero_duration_rows = int(
        (combined_df["duration_seconds"] == 0).sum()
    )

    # Save the combined dataset.
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("\nCOMBINATION RESULTS")
    print("-" * 40)
    print(f"Collection sessions: {len(csv_files)}")
    print(f"Total flow records: {len(combined_df)}")
    print(f"Number of features: {len(combined_df.columns)}")
    print(f"Missing values: {missing_values}")
    print(f"Infinite values: {infinite_values}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Zero-duration flows: {zero_duration_rows}")

    print("\nProtocol distribution:")
    print(combined_df["protocol"].value_counts())

    print(f"\nCombined dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
