# Import pandas and Path
import pandas as pd
from pathlib import Path

# Import our prediction function
from prediction_engine import predict_threat


# Define the processed test-data folder
processed_folder = Path("dataset/processed")


# Load the test features
X_test = pd.read_csv(processed_folder / "X_test.csv")


# Take one real network-flow record
# Load the test labels so we can locate a BENIGN sample
# Select the first test record for malicious traffic testing
sample = X_test.iloc[0].to_dict()

# TEST ONLY:
# Remove one required feature to check input validation
#sample.pop("Rate")

print("Using test record at index: 0")


# Send the sample to our prediction engine
result = predict_threat(sample)


# Display the result
print("=== THREAT PREDICTION TEST ===")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("Threat Score:", result["threat_score"])
print("Threat Level:", result["threat_level"])



# Display the selected test record as JSON
import json

print("\n=== JSON FOR FASTAPI TEST ===")
print(
    json.dumps(
        {"features": sample},
        indent=2
    )
)