# Import FastAPI components
from fastapi import FastAPI, HTTPException

# Import Pydantic for request/response validation
from pydantic import BaseModel

# Import typing support
from typing import Dict

# Import the prediction engine
from src.prediction_engine import predict_threat


# Create the FastAPI application
app = FastAPI(
    title="AI Driven Cyber Threat Intelligence API",
    description="Backend API for network intrusion detection and threat scoring",
    version="1.0.0"
)


# Define the expected structure of incoming prediction data
class PredictionRequest(BaseModel):
    # Every feature name must be a string
    # Every feature value must be numeric
    features: Dict[str, float]


# Define the structure of a successful prediction response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    threat_score: float
    threat_level: str


# Simple health-check endpoint
@app.get("/health")
def health_check():
    """
    Confirm that the API server is running correctly.
    """

    return {
        "status": "healthy",
        "message": "Cyber Threat Intelligence API is running"
    }


# Machine-learning prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {
            "description": "Invalid or incomplete network feature data"
        }
    }
)
def predict(request: PredictionRequest):
    """
    Accept network-flow features and return:

    - predicted attack class
    - model confidence
    - threat score
    - threat level
    """

    try:
        # Send the network features to the prediction engine
        result = predict_threat(request.features)

        # Return the successful prediction
        return result

    except ValueError as error:
        # Convert prediction-engine validation errors
        # into a clean HTTP 400 response
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )