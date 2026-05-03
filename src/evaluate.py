# src/evaluate.py

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula métricas básicas para un modelo de regresión.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    mape = np.mean(
        np.abs((y_true_array - y_pred_array) / y_true_array)
    ) * 100

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Imprime métricas de evaluación en consola.
    """

    print("Métricas de evaluación:")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"MSE:   {metrics['mse']:.4f}")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print(f"R2:    {metrics['r2']:.4f}")