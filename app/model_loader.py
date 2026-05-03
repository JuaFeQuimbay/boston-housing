# app/model_loader.py

from pathlib import Path

import joblib


MODEL_PATH = Path("models/model.pkl")


def load_model():
    """
    Carga el modelo entrenado desde disco.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero: python -m src.train"
        )

    return joblib.load(MODEL_PATH)
