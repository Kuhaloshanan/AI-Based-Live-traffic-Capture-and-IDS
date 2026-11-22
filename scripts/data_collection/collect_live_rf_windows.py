
"""Collect live packet windows for Random Forest validation."""

from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
from scapy.all import sniff

from src.live_rf_window_processor import (
    LiveRFWindowProcessor,
    RF_FEATURE_COLUMNS
)


INTERFACE = "Wi-Fi"
CAPTURE_DURATION = 600

OUTPUT_DIR = Path("dataset/live_rf_validation")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = LiveRFWindowProcessor(
        window_sizes=(10, 100),
        idle_timeout=30
    )

    collected_rows = []

    print("\n========== LIVE RF FEATURE COLLECTION ==========")
    print(f"Interface: {INTERFACE}")
    print(f"Duration: {CAPTURE_DURATION} seconds")
    print("Browse websites or play YouTube during collection.")
    print("Collecting live packet windows...\n")

    def handle_packet(packet):

        completed = processor.process_packet(packet)

        for result in completed:

            row = {
                "window_size": result["window_size"],
                **result["features"]
            }

            collected_rows.append(row)

            if len(collected_rows) % 25 == 0:
                print(
                    f"Completed windows: {len(collected_rows)}"
                )

    try:

        sniff(
            iface=INTERFACE,
            prn=handle_packet,
            store=False,
            timeout=CAPTURE_DURATION
        )

    except KeyboardInterrupt:
        print("\nCapture stopped manually.")

    if not collected_rows:
        print("No complete packet windows were collected.")
        return

    columns = ["window_size"] + RF_FEATURE_COLUMNS

    dataframe = pd.DataFrame(
        collected_rows,
        columns=columns
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = OUTPUT_DIR / (
        f"live_rf_windows_{timestamp}.csv"
    )

    dataframe.to_csv(output_file, index=False)

    counts = Counter(dataframe["window_size"])

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Total windows: {len(dataframe)}")
    print(f"10-packet windows: {counts.get(10, 0)}")
    print(f"100-packet windows: {counts.get(100, 0)}")
    print(f"Features per window: {len(RF_FEATURE_COLUMNS)}")
    print(f"Saved file: {output_file}")
    print("========================================")


if __name__ == "__main__":
    main()
