
"""
Inspect the combined normal-traffic baseline.

This script helps us understand feature distributions
before training the Isolation Forest model.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "live_baseline"
    / "normal_traffic_combined.csv"
)


def main():
    # Load the combined baseline dataset.
    df = pd.read_csv(DATASET_FILE)

    print("BASELINE DATASET OVERVIEW")
    print("-" * 50)

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Protocol is categorical; inspect it separately.
    print("\nPROTOCOL DISTRIBUTION")
    print("-" * 50)
    print(df["protocol"].value_counts())

    # Select numerical features only.
    numeric_df = df.select_dtypes(include="number")

    # Display important statistical measurements.
    print("\nNUMERICAL FEATURE STATISTICS")
    print("-" * 50)
    print(
        numeric_df.describe().T[
            ["mean", "std", "min", "50%", "max"]
        ].round(3).to_string()
    )

    # Skewness indicates whether values are distributed
    # symmetrically or have a long tail.
    print("\nFEATURE SKEWNESS")
    print("-" * 50)
    print(
        numeric_df.skew()
        .sort_values(ascending=False)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
