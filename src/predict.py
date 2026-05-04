# src/predict.py

from pathlib import Path
from typing import Dict, List, Union

import joblib
import pandas as pd


# 🔥 Ruta robusta independiente del working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """
    Carga el modelo entrenado desde disco.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta primero: python -m src.train"
        )

    return joblib.load(model_path)


def predict_single(
    input_data: Dict[str, Union[int, float]],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> float:
    """
    Realiza una predicción para un único registro.
    """

    model = load_model(model_path)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return float(prediction)


def predict_batch(
    input_data: List[Dict[str, Union[int, float]]],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> List[float]:
    """
    Realiza predicciones para múltiples registros.
    """

    model = load_model(model_path)

    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)

    return [float(pred) for pred in predictions]