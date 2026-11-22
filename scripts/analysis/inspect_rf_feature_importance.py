
"""Inspect feature importance in the existing Random Forest."""

import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_improved_ciciot2023.pkl"


def main():

    model = joblib.load(MODEL_PATH)

    # Get feature names and their trained importance values.
    importance_df = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    })

    # Sort from most important to least important.
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====\n")

    print(
        importance_df.to_string(
            index=False,
            formatters={"importance": "{:.6f}".format}
        )
    )

    print("\n===== NUMBER FEATURE IMPORTANCE =====")

    number_importance = importance_df[
        importance_df["feature"] == "Number"
    ]

    print(number_importance.to_string(index=False))


if __name__ == "__main__":
    main()
