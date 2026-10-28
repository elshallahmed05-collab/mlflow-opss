
"""FastAPI service for churn predictions using an MLflow model."""

import os
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using a model registered in MLflow.",
    version="1.0.0",
)

MODEL_NAME = os.getenv("MODEL_NAME", "ChurnClassifier")
MODEL = None


class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    contract_type: str
    internet_service: str


def load_model():
    """Load the latest registered model from MLflow."""
    global MODEL

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    for attempt in range(30):
        try:
            versions = client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )

            if versions:
                latest = max(
                    versions,
                    key=lambda version: int(version.version),
                )
                MODEL = mlflow.pyfunc.load_model(
                    f"models:/{MODEL_NAME}/{latest.version}"
                )
                print("MLflow model loaded successfully.")
                return

        except Exception as exc:
            print(f"Waiting for model: {exc}")

        time.sleep(2)

    print("Warning: model was not loaded.")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
def predict(payload: PredictionInput):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MLflow and trainer logs.",
        )

    row = pd.DataFrame([payload.model_dump()])
    prediction = int(MODEL.predict(row)[0])

    probability = None
    try:
        probability = float(MODEL.predict_proba(row)[0][1])
    except (AttributeError, IndexError, TypeError):
        pass

    return {
        "prediction": prediction,
        "label": "churn" if prediction == 1 else "no_churn",
        "probability": probability,
    }
