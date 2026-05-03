# app/main.py

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_model
from app.schemas import HousingFeatures, PredictionResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"

app = FastAPI(
    title="Housing Price Prediction API",
    description="API REST para predecir precios de viviendas usando Boston Housing.",
    version="1.0.0",
)

model = load_model()


@app.get("/health")
def health_check():
    """
    Endpoint para validar que la API está activa.
    """

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    """
    Endpoint de inferencia para predecir el precio de una vivienda.
    """

    start_time = time.time()

    try:
        input_data = features.model_dump()
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            {
                "event": "prediction",
                "prediction": float(prediction),
                "latency_ms": latency_ms,
                "model_version": MODEL_VERSION,
            }
        )

        return PredictionResponse(
            prediction=float(prediction),
            model_version=MODEL_VERSION,
        )

    except Exception as error:
        logger.exception("Error during prediction")

        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(error)}",
        )