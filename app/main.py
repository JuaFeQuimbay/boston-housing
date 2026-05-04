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

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Orden fijo de columnas: el ColumnTransformer fue ajustado con estos nombres.
# Mantenerlo precomputado evita rehacer el cálculo en cada request.
FEATURE_ORDER = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

app = FastAPI(
    title="Housing Price Prediction API",
    description="API REST para predecir precios de viviendas usando Boston Housing.",
    version="1.0.0",
)

model = load_model()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    start_time = time.perf_counter()

    try:
        feature_dict = features.model_dump()
        # Construir el DataFrame desde un array 2D con orden conocido es
        # significativamente más rápido que pd.DataFrame([dict]).
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
