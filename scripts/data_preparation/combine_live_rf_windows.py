
"""Combine timestamped live RF window captures into one dataset."""

from pathlib import Path
import re
import pandas as pd

# Folder containing the individual live packet-window captures.
DATA_DIR = Path("dataset/live_rf_validation")

OUTPUT_FILE = DATA_DIR / "live_rf_windows_combined.csv"

# The 39 features expected by the existing CICIoT2023 Random Forest.
EXPECTED_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Variance"
]

def main():
    # Select timestamped captures only, avoiding the combined output.
    capture_files = sorted(
        path for path in DATA_DIR.glob("live_rf_windows_*.csv")
        if re.fullmatch(
            r"live_rf_windows_\d{8}_\d{6}\.csv",
            path.name
        )
    )

    if not capture_files:
        raise FileNotFoundError("No timestamped live capture files found.")

    dataframes = []

    for file_path in capture_files:
        print(f"Reading: {file_path.name}")

        df = pd.read_csv(file_path)

        required = set(EXPECTED_FEATURES + ["window_size"])
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: {sorted(missing)}"
            )

        # Preserve the source filename for traceability.
        df["source_file"] = file_path.name
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    # Check the combined dataset before saving.
    numeric_data = combined[EXPECTED_FEATURES + ["window_size"]]
    numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce")

    if numeric_data.isna().any().any():
        raise ValueError("Missing or non-numeric values found.")

    if not numeric_data.map(
        lambda column: column if isinstance(column, bool) else column
    ).shape:
        raise ValueError("Unexpected dataset shape.")

    combined[EXPECTED_FEATURES + ["window_size"]] = numeric_data

    if not numeric_data.apply(
        lambda column: column.map(lambda value: abs(value) != float("inf"))
    ).all().all():
        raise ValueError("Infinite values found.")

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\n========== COMBINATION COMPLETE ==========")
    print("Capture files:", len(capture_files))
    print("Total windows:", len(combined))
    print("\nWindow-size distribution:")
    print(combined["window_size"].value_counts().sort_index())
    print("\nSource-file distribution:")
    print(combined["source_file"].value_counts())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
