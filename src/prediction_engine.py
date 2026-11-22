# Import required libraries
import joblib
import pandas as pd
from pathlib import Path



# MODEL PATH SETUP
# ---------------------------------------------------------

# Find the main project directory.
# prediction_engine.py is inside src/,
# so parent.parent gives the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Build the full path to the selected improved model.
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_improved_ciciot2023.pkl"
)


# Load the trained Random Forest model.
model = joblib.load(MODEL_PATH)



# THREAT SCORING
# ---------------------------------------------------------

def calculate_threat_score(predicted_class, confidence):
    """
    Calculate a threat score between 0 and 100.

    The score considers:
    1. The predicted attack category
    2. The model prediction confidence

    Higher-risk attack families receive a higher
    base severity score.
    """

    # BENIGN traffic is handled separately.
    # Higher confidence in BENIGN means a lower threat score.
    if predicted_class == "BENIGN":
        score = (1 - confidence) * 20

        # Ensure the score remains between 0 and 100.
        score = max(0, min(100, score))

        return round(score, 2)



    # ATTACK BASE SEVERITY
    # -----------------------------------------------------

    # Distributed Denial-of-Service attacks
    if predicted_class.startswith("DDOS"):
        base_severity = 75

    # Denial-of-Service attacks
    elif predicted_class.startswith("DOS"):
        base_severity = 70

    # Mirai botnet-related attacks
    elif predicted_class.startswith("MIRAI"):
        base_severity = 80

    # High-risk application/system attacks
    elif predicted_class in [
        "BACKDOOR_MALWARE",
        "COMMANDINJECTION",
        "SQLINJECTION",
        "UPLOADING_ATTACK"
    ]:
        base_severity = 85

    # Spoofing / interception attacks
    elif predicted_class in [
        "MITM-ARPSPOOFING",
        "DNS_SPOOFING"
    ]:
        base_severity = 70

    # Reconnaissance attacks
    elif predicted_class.startswith("RECON"):
        base_severity = 45

    # Vulnerability scanning
    elif predicted_class == "VULNERABILITYSCAN":
        base_severity = 50

    # Web / brute-force attacks
    elif predicted_class in [
        "BROWSERHIJACKING",
        "XSS",
        "DICTIONARYBRUTEFORCE"
    ]:
        base_severity = 60

    # Default severity for any other malicious class
    else:
        base_severity = 50



    # FINAL THREAT SCORE
    # -----------------------------------------------------

    # Convert model confidence from 0-1 into 0-100.
    confidence_score = confidence * 100

    # 70% of the score comes from attack severity.
    # 30% comes from model confidence.
    score = (
        0.70 * base_severity
        + 0.30 * confidence_score
    )

    # Ensure the score remains between 0 and 100.
    score = max(0, min(100, score))

    return round(score, 2)



# THREAT LEVEL
# ---------------------------------------------------------

def get_threat_level(score):
    """
    Convert the numerical threat score into
    a human-readable threat level.
    """

    if score < 25:
        return "LOW"

    elif score < 50:
        return "MEDIUM"

    elif score < 75:
        return "HIGH"

    else:
        return "CRITICAL"



# PREDICTION ENGINE
# ---------------------------------------------------------

def predict_threat(input_data):
    """
    Accept network-flow features and perform:

    1. Feature validation
    2. Random Forest prediction
    3. Prediction confidence calculation
    4. Threat score calculation
    5. Threat level classification

    Parameters
    ----------
    input_data : dict
        Dictionary containing the 39 network-flow features.

    Returns
    -------
    dict
        Prediction result containing:
        - prediction
        - confidence
        - threat_score
        - threat_level
    """

    # Get the exact feature names expected by the trained model.
    required_features = list(model.feature_names_in_)



    # CHECK FOR MISSING FEATURES
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_data
    ]

    # Stop prediction if required features are missing.
    if missing_features:
        raise ValueError(
            "Missing required network features: "
            + ", ".join(missing_features)
        )



    # CHECK FOR EXTRA FEATURES
    # -----------------------------------------------------

    extra_features = [
        feature
        for feature in input_data
        if feature not in required_features
    ]

    # Stop prediction if unexpected features are provided.
    if extra_features:
        raise ValueError(
            "Unexpected network features: "
            + ", ".join(extra_features)
        )



    # PREPARE INPUT FOR THE MODEL
    # -----------------------------------------------------

    # Convert the input dictionary into a one-row DataFrame.
    df = pd.DataFrame([input_data])

    # Reorder the columns so they match the exact order
    # used during Random Forest training.
    df = df[model.feature_names_in_]



    # MODEL PREDICTION
    # -----------------------------------------------------

    # Predict the network traffic class.
    prediction = model.predict(df)[0]

    # Get class probabilities from the Random Forest.
    probabilities = model.predict_proba(df)[0]

    # The highest probability is used as model confidence.
    confidence = probabilities.max()



    # THREAT ANALYSIS
    # -----------------------------------------------------

    # Calculate the threat score.
    threat_score = calculate_threat_score(
        prediction,
        confidence
    )

    # Convert the score into a threat level.
    threat_level = get_threat_level(
        threat_score
    )



    # RETURN RESULT


    return {
        "prediction": str(prediction),
        "confidence": round(float(confidence), 4),
        "threat_score": threat_score,
        "threat_level": threat_level
    }



# DIRECT FILE TEST


# This runs only when prediction_engine.py itself is executed.
# It will NOT run when FastAPI imports this module.
if __name__ == "__main__":
    print("Prediction engine loaded successfully.")
    print("Model loaded from:", MODEL_PATH)
    print("Number of required features:", len(model.feature_names_in_))