
"""
Inspect packet counts in the CICIoT2023 training CSV.

This script reads only the Number and Label columns.
It does not modify any dataset files.
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd


DATASET_PATH = Path("dataset/MERGED_CSV/Merged01.csv")


def main():

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    # Store packet-count frequencies separately for each class.
    results = defaultdict(lambda: defaultdict(int))

    total_rows = 0

    # Read only two columns to reduce memory usage.
    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=["Number", "Label"],
        chunksize=100_000
    ):

        total_rows += len(chunk)

        # Count how often each packet count occurs per label.
        grouped = chunk.groupby(["Label", "Number"]).size()

        for (label, number), count in grouped.items():
            results[str(label)][number] += int(count)

    print("\n========== PACKET WINDOW AUDIT ==========")
    print(f"Total records inspected: {total_rows:,}")

    for label, frequencies in sorted(results.items()):

        print(f"\nClass: {label}")

        for number, count in sorted(frequencies.items()):
            print(f"  Number={number}: {count:,} records")

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
