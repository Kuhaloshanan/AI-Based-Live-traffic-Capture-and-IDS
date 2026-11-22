
"""
Live Network Baseline Collector

Captures ordinary network traffic, processes packets into flows,
and saves flow statistics to a CSV file.

The collected data will later be used to train an
Isolation Forest anomaly detection model.
"""

import csv
import time

from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

from scapy.all import AsyncSniffer

from src.flow_processor import FlowProcessor


# CONFIGURATION


INTERFACE_NAME = "Wi-Fi"

# Start with a 2-minute test collection.
# We will collect a longer baseline after verifying this works.
COLLECTION_DURATION_SECONDS = 600 #10 Minutes

IDLE_TIMEOUT = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "live_baseline"



# FEATURES TO SAVE

# These are the flow statistics we want to preserve.
# Protocol is categorical; the remaining fields are numerical.

CSV_COLUMNS = [
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
    "byte_rate"
]



# INITIALIZE COMPONENTS


packet_queue = Queue()

flow_processor = FlowProcessor(
    idle_timeout=IDLE_TIMEOUT
)



# SAVE COMPLETED FLOWS


def save_completed_flows(flows, writer, output_file):
    """
    Save completed network flows to the CSV file.

    Only selected flow statistics are stored.
    IP addresses and port numbers are excluded.
    """

    saved_count = 0

    for flow in flows:

        # Extract only the required columns.
        row = {
            column: flow[column]
            for column in CSV_COLUMNS
        }

        writer.writerow(row)

        saved_count += 1

    # Ensure newly collected records are written to disk.
    if saved_count > 0:
        output_file.flush()

    return saved_count



# START BASELINE COLLECTION


def start_baseline_collection():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create a unique filename for each collection session.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = OUTPUT_DIR / f"normal_traffic_{timestamp}.csv"

    print("=" * 65)
    print("AI CYBER THREAT INTELLIGENCE SYSTEM")
    print("LIVE NORMAL TRAFFIC BASELINE COLLECTION")
    print("=" * 65)

    print(f"Network interface: {INTERFACE_NAME}")

    print(
        f"Collection duration: "
        f"{COLLECTION_DURATION_SECONDS} seconds"
    )

    print(f"Output file: {output_path}")

    print("\nKeep your network activity ordinary.")
    print("Do not run attack simulations or network scans.")
    print("Press CTRL + C to stop early.")

    print("-" * 65)

    # Prepare the CSV file.
    with open(
        output_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()

        # Create the background packet sniffer.
        sniffer = AsyncSniffer(
            iface=INTERFACE_NAME,
            prn=packet_queue.put,
            store=False
        )

        saved_flows = 0
        capture_started = False

        try:

            sniffer.start()
            capture_started = True

            print("\nPacket capture started successfully.")
            print("Collecting normal network traffic...\n")

            start_time = time.monotonic()

            while (
                time.monotonic() - start_time
                < COLLECTION_DURATION_SECONDS
            ):

                try:

                    # Retrieve one captured packet.
                    packet = packet_queue.get(timeout=1)

                    # Process the packet into a network flow.
                    completed = flow_processor.process_packet(packet)

                    saved_flows += save_completed_flows(
                        completed,
                        writer,
                        csv_file
                    )

                except Empty:
                    pass

                # Expire flows that have been inactive.
                expired = flow_processor.expire_flows(
                    time.time()
                )

                saved_flows += save_completed_flows(
                    expired,
                    writer,
                    csv_file
                )

        except KeyboardInterrupt:

            print("\nCollection interrupted by user.")

        finally:

            # Stop packet capture safely.
            if capture_started and sniffer.running:
                sniffer.stop()

            # Process packets still waiting in the queue.
            while True:

                try:

                    packet = packet_queue.get_nowait()

                    completed = flow_processor.process_packet(packet)

                    saved_flows += save_completed_flows(
                        completed,
                        writer,
                        csv_file
                    )

                except Empty:
                    break

            # Save all remaining active flows.
            remaining = flow_processor.flush()

            saved_flows += save_completed_flows(
                remaining,
                writer,
                csv_file
            )

            print("\n" + "=" * 65)
            print("BASELINE COLLECTION FINISHED")
            print("=" * 65)

            print(f"Total saved flows: {saved_flows}")
            print(f"CSV location: {output_path}")

            print("=" * 65)

    return output_path



# PROGRAM ENTRY POINT


if __name__ == "__main__":
    start_baseline_collection()
