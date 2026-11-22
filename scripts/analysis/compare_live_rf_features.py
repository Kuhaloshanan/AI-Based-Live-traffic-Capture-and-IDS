
"""
Compare live packet-window features against CICIoT2023.

This is a distribution audit, not a model accuracy test.
It does not modify either dataset or the trained model.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path("dataset/MERGED_CSV/Merged01.csv")

LIVE_PATH = Path(
    "dataset/live_rf_validation/"
    "live_rf_windows_20261122_002930.csv"
)

OUTPUT_PATH = Path(
    "dataset/live_rf_validation/"
    "feature_distribution_comparison.csv"
)

FEATURES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

IMPORTANT_FEATURES = [
    "Header_Length",
    "Tot sum",
    "Tot size",
    "AVG",
    "Max",
    "Rate",
    "IAT",
    "Variance",
    "Std",
    "Time_To_Live",
    "Protocol Type",
    "ack_count",
    "TCP",
    "UDP",
    "HTTPS",
]


def get_statistics(series):

    values = pd.to_numeric(
        series, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return None

    return {
        "p05": values.quantile(0.05),
        "median": values.median(),
        "p95": values.quantile(0.95),
    }


def main():

    if not TRAIN_PATH.exists():
        print(f"Training dataset not found: {TRAIN_PATH}")
        return

    if not LIVE_PATH.exists():
        print(f"Live dataset not found: {LIVE_PATH}")
        print("Copy the collected CSV into dataset/live_rf_validation.")
        return

    print("Loading CICIoT2023 training records...")

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["Label"] + FEATURES
    )

    for feature in FEATURES:
        train[feature] = pd.to_numeric(
            train[feature], errors="coerce"
        )

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.dropna(subset=FEATURES, inplace=True)

    print("Loading live packet windows...")

    live = pd.read_csv(LIVE_PATH)

    print("\nTraining records:", len(train))
    print("Live records:", len(live))

    print("\nTraining packet-count distribution:")
    print(train["Number"].value_counts().sort_index())

    print("\nLive packet-window distribution:")
    print(live["window_size"].value_counts().sort_index())

    results = []

    for window_size in (10, 100):

        training_group = train[
            train["Number"] == window_size
        ]

        live_group = live[
            live["window_size"] == window_size
        ]

        print(f"\n========== {window_size}-PACKET WINDOWS ==========")
        print("Training records:", len(training_group))
        print("Live records:", len(live_group))

        if training_group.empty or live_group.empty:
            print("Not enough records for comparison.")
            continue

        for feature in FEATURES:

            training_stats = get_statistics(
                training_group[feature]
            )

            live_stats = get_statistics(
                live_group[feature]
            )

            if training_stats is None or live_stats is None:
                continue

            row = {
                "window_size": window_size,
                "feature": feature,
                "training_p05": training_stats["p05"],
                "live_p05": live_stats["p05"],
                "training_median": training_stats["median"],
                "live_median": live_stats["median"],
                "training_p95": training_stats["p95"],
                "live_p95": live_stats["p95"],
            }

            results.append(row)

            if feature in IMPORTANT_FEATURES:
                print(
                    f"\n{feature}"
                    f"\n  Training median: "
                    f"{training_stats['median']:.4f}"
                    f"\n  Live median: "
                    f"{live_stats['median']:.4f}"
                    f"\n  Training P95: "
                    f"{training_stats['p95']:.4f}"
                    f"\n  Live P95: "
                    f"{live_stats['p95']:.4f}"
                )

    if results:

        report = pd.DataFrame(results)

        OUTPUT_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )

        report.to_csv(OUTPUT_PATH, index=False)

        print("\nComparison report saved:")
        print(OUTPUT_PATH)

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
