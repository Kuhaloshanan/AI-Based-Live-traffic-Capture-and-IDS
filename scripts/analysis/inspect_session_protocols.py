
"""
Inspect protocol representation in each baseline session.

We need to understand whether rare protocols are absent
or underrepresented during model training.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "live_baseline"

SESSION_FILES = [
    DATASET_DIR / "normal_traffic_20261116_231038.csv",
    DATASET_DIR / "normal_traffic_20261116_231848.csv",
]


def main():
    print("PROTOCOL REPRESENTATION BY SESSION")
    print("=" * 60)

    for session_file in SESSION_FILES:

        df = pd.read_csv(session_file)

        print(f"\nSession: {session_file.name}")
        print("-" * 60)

        counts = df["protocol"].value_counts()

        percentages = (
            df["protocol"].value_counts(normalize=True) * 100
        )

        summary = pd.DataFrame({
            "flow_count": counts,
            "percentage": percentages.round(2)
        })

        print(summary.to_string())

        print(f"\nTotal flows: {len(df)}")


if __name__ == "__main__":
    main()
