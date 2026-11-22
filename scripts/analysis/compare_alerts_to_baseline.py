
"""
Compare recent anomalous flows against the collected
normal-traffic baseline.

This helps us understand how unusual the alert traffic
is compared with previously observed normal activity.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASELINE_FILE = (
    PROJECT_ROOT / "dataset" / "live_baseline"
    / "normal_traffic_combined.csv"
)

ALERT_FILE = (
    PROJECT_ROOT / "logs" / "security_alerts.csv"
)

# Load the normal baseline and most recent 42 alerts.
baseline = pd.read_csv(BASELINE_FILE)
alerts = pd.read_csv(ALERT_FILE).tail(42).copy()

# Calculate rates using the same definitions as the flow processor.
alerts["packet_rate"] = (
    alerts["packet_count"] / alerts["duration_seconds"].replace(0, float("nan"))
)

alerts["byte_rate"] = (
    alerts["total_bytes"] / alerts["duration_seconds"].replace(0, float("nan"))
)

# Compare three traffic characteristics.
metrics = [
    "duration_seconds",
    "packet_rate",
    "byte_rate",
]

print("\nNORMAL BASELINE PERCENTILES")
print("=" * 65)

for metric in metrics:

    values = baseline[metric].dropna()

    print(f"\n{metric}")
    print(values.quantile([0.50, 0.90, 0.95, 0.99]).round(3))

# Calculate where each alert sits in the baseline distribution.
def percentile_rank(value, baseline_values):

    if pd.isna(value):
        return float("nan")

    return (
        (baseline_values <= value).mean() * 100
    )


# Display the ten highest-scoring alerts.
print("\nTOP 10 ALERTS: BASELINE COMPARISON")
print("=" * 100)

top_alerts = alerts.nlargest(10, "anomaly_score")

for _, alert in top_alerts.iterrows():

    print(f"\nProtocol: {alert['protocol']}")
    print(f"Anomaly score: {alert['anomaly_score']:.6f}")

    for metric in metrics:

        baseline_values = baseline[metric].dropna()

        percentile = percentile_rank(
            alert[metric],
            baseline_values
        )

        print(
            f"{metric}: {alert[metric]:.3f} "
            f"(baseline percentile: {percentile:.2f}%)"
        )


# SUMMARY OF ALL ALERTS AGAINST BASELINE THRESHOLDS


print("\nALL ALERTS: BASELINE THRESHOLD ANALYSIS")
print("=" * 75)

for metric in metrics:

    baseline_values = baseline[metric].dropna()

    # Calculate thresholds from previously collected normal traffic.
    p05 = baseline_values.quantile(0.05)
    p95 = baseline_values.quantile(0.95)
    p99 = baseline_values.quantile(0.99)

    # Count alerts outside the normal baseline thresholds.
    below_p05 = (alerts[metric] < p05).sum()
    above_p95 = (alerts[metric] > p95).sum()
    above_p99 = (alerts[metric] > p99).sum()

    print(f"\nMetric: {metric}")
    print(f"Baseline 5th percentile:  {p05:.3f}")
    print(f"Baseline 95th percentile: {p95:.3f}")
    print(f"Baseline 99th percentile: {p99:.3f}")

    print(f"Alerts below 5th percentile:  {below_p05}")
    print(f"Alerts above 95th percentile: {above_p95}")
    print(f"Alerts above 99th percentile: {above_p99}")




# ANALYSE COMBINATIONS OF UNUSUAL TRAFFIC CHARACTERISTICS


print("\nALERT CHARACTERISTIC COMBINATIONS")
print("=" * 75)

# Calculate the 95th percentile thresholds.
duration_p95 = baseline["duration_seconds"].quantile(0.95)
packet_rate_p95 = baseline["packet_rate"].quantile(0.95)
byte_rate_p95 = baseline["byte_rate"].quantile(0.95)

# Identify which alerts exceed each threshold.
flags = pd.DataFrame({
    "Long_Duration": alerts["duration_seconds"] > duration_p95,
    "High_Packet_Rate": alerts["packet_rate"] > packet_rate_p95,
    "High_Byte_Rate": alerts["byte_rate"] > byte_rate_p95,
})

# Count each unique combination.
combination_counts = (
    flags.value_counts()
    .reset_index(name="Alert_Count")
)

print("\nCombinations using 95th percentile thresholds:")
print(combination_counts.to_string(index=False))

# Count how many alerts exhibit multiple unusual characteristics.
multiple_flags = flags.sum(axis=1) >= 2
all_three_flags = flags.all(axis=1)

print("\nAdditional findings:")
print(
    "Alerts with at least 2 unusual characteristics:",
    int(multiple_flags.sum())
)

print(
    "Alerts with all 3 unusual characteristics:",
    int(all_three_flags.sum())
)




# ANALYSE COMBINATIONS OF UNUSUAL TRAFFIC CHARACTERISTICS


print("\nALERT CHARACTERISTIC COMBINATIONS")
print("=" * 75)

# Use the 95th percentile of normal traffic as reference thresholds.
duration_p95 = baseline["duration_seconds"].quantile(0.95)
packet_rate_p95 = baseline["packet_rate"].quantile(0.95)
byte_rate_p95 = baseline["byte_rate"].quantile(0.95)

# Identify unusual characteristics for each alert.
flags = pd.DataFrame({
    "Long_Duration": alerts["duration_seconds"] > duration_p95,
    "High_Packet_Rate": alerts["packet_rate"] > packet_rate_p95,
    "High_Byte_Rate": alerts["byte_rate"] > byte_rate_p95,
})

# Give each alert a readable combination label.
def describe_pattern(row):
    patterns = [
        name for name in flags.columns if row[name]
    ]
    return " + ".join(patterns) if patterns else "No extreme metric"

flags["Pattern"] = flags.apply(describe_pattern, axis=1)

print("\nCombination counts:")
print(
    flags["Pattern"]
    .value_counts()
    .rename_axis("Pattern")
    .reset_index(name="Alert_Count")
    .to_string(index=False)
)

# Count alerts with multiple unusual characteristics.
multiple_flags = flags.iloc[:, :3].sum(axis=1) >= 2
all_three_flags = flags.iloc[:, :3].all(axis=1)

print("\nSUMMARY")
print("Alerts with at least 2 unusual characteristics:",
      int(multiple_flags.sum()))

print("Alerts with all 3 unusual characteristics:",
      int(all_three_flags.sum()))
