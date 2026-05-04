# app/main.py

import logging
import os
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_model
from app.schemas import HousingFeatures, PredictionResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Permite override manual; si no se setea, usa la versión derivada del run de MLflow
MODEL_VERSION_OVERRIDE = os.getenv("MODEL_VERSION")

app = FastAPI(
    title="Housing Price Prediction API",
    description="API REST para predecir precios de viviendas usando Boston Housing.",
    version="1.0.0",
)

bundle = load_model()
model = bundle.model
MODEL_VERSION = MODEL_VERSION_OVERRIDE or bundle.version
FEATURE_ORDER = bundle.feature_order


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "trained_at": bundle.metadata.get("trained_at"),
        "git_sha": bundle.metadata.get("git_sha"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    start_time = time.perf_counter()

    try:
        feature_dict = features.model_dump()
        row = np.array([[feature_dict[name] for name in FEATURE_ORDER]], dtype=np.float64)
        input_df = pd.DataFrame(row, columns=FEATURE_ORDER)

        prediction = float(model.predict(input_df)[0])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            {
                "event": "prediction",
                "prediction": prediction,
                "latency_ms": latency_ms,
                "model_version": MODEL_VERSION,
            }
        )

        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION,
        )

    except Exception as error:
        logger.exception("Error during prediction")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {error}",
        )
