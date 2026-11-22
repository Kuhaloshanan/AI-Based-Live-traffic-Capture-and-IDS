
"""
Audit the compatibility between:
1. The existing CICIoT2023 Random Forest model.
2. The live network FlowProcessor.

This script only inspects existing files. It does not modify the model.
"""

import inspect
import joblib
from pathlib import Path

from src.flow_processor import FlowProcessor


# Existing trained Random Forest model
MODEL_PATH = Path("models/random_forest_improved_ciciot2023.pkl")


def main():
    print("\n========== RANDOM FOREST / LIVE FEATURE AUDIT ==========\n")

    # Load the existing trained model
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    # Display the exact features expected by Random Forest
    rf_features = list(model.feature_names_in_)

    print("1. RANDOM FOREST EXPECTED FEATURES")
    print("-" * 55)

    for index, feature in enumerate(rf_features, start=1):
        print(f"{index:02d}. {feature}")

    print(f"\nTotal Random Forest features: {len(rf_features)}")

    # Inspect the actual live flow finalization logic
    print("\n\n2. LIVE FLOW PROCESSOR")
    print("-" * 55)

    method = getattr(FlowProcessor, "finalize_flow", None)

    if method is None:
        print("Could not find FlowProcessor.finalize_flow")
        return

    print("\nActual finalize_flow implementation:\n")
    print(inspect.getsource(method))

    print("\n========== AUDIT COMPLETE ==========\n")


if __name__ == "__main__":
    main()
