# src/train.py

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data, split_data
from src.evaluate import evaluate_regression_model, print_metrics
from src.features import build_preprocessing_pipeline


RANDOM_STATE = 42

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

REPORTS_DIR = Path("reports")
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PERMUTATION_IMPORTANCE_PATH = REPORTS_DIR / "permutation_importance.csv"


def _resolve_git_sha() -> str:
    """Devuelve el SHA del commit actual; 'unknown' si no hay repo o git falla."""

    if env_sha := os.getenv("GITHUB_SHA"):
        return env_sha[:7]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def train_model():
    """
    Entrena un modelo de regresión para predecir el valor mediano de viviendas
    usando el dataset Boston Housing.

    Incluye:
    - Preprocesamiento
    - Búsqueda de hiperparámetros
    - Evaluación
    - Persistencia del modelo
    - Registro de métricas y artefactos en MLflow
    - Explicabilidad básica mediante feature importance
    """

    df = load_data()
    X, y = split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessing_pipeline(X_train)

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_experiment("housing-price-regression")

    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_mae = abs(grid_search.best_score_)

        y_pred = best_pipeline.predict(X_test)
        metrics = evaluate_regression_model(y_test, y_pred)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("optimization_metric", "MAE")

        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("best_cv_mae", best_cv_mae)
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mape", metrics["mape"])
        mlflow.log_metric("r2", metrics["r2"])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_pipeline, MODEL_PATH)

        feature_names = X_train.columns.tolist()
        trained_model = best_pipeline.named_steps["model"]

        feature_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": trained_model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)

        feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        permutation_result = permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        permutation_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": permutation_result.importances_mean,
                "importance_std": permutation_result.importances_std,
            }
        ).sort_values(by="importance_mean", ascending=False)

        permutation_importance_df.to_csv(
            PERMUTATION_IMPORTANCE_PATH,
            index=False,
        )

        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))
        mlflow.log_artifact(str(PERMUTATION_IMPORTANCE_PATH))

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
        )

        run_id = mlflow.active_run().info.run_id
        metadata = {
            "model_version": run_id[:8],
            "mlflow_run_id": run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _resolve_git_sha(),
            "model_type": "RandomForestRegressor",
            "best_params": best_params,
            "metrics": {
                "cv_mae": round(best_cv_mae, 4),
                "test_mae": round(metrics["mae"], 4),
                "test_rmse": round(metrics["rmse"], 4),
                "test_mape": round(metrics["mape"], 4),
                "test_r2": round(metrics["r2"], 4),
            },
            "feature_order": feature_names,
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(METADATA_PATH))

        print("Modelo entrenado correctamente")
        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Metadata guardada en: {METADATA_PATH}")
        print(f"Versión del modelo: {metadata['model_version']}")
        print(f"Mejores hiperparámetros: {best_params}")
        print(f"Mejor MAE promedio en CV: {best_cv_mae:.4f}")
        print_metrics(metrics)
        print(f"Feature importance guardada en: {FEATURE_IMPORTANCE_PATH}")
        print(f"Permutation importance guardada en: {PERMUTATION_IMPORTANCE_PATH}")


if __name__ == "__main__":
    train_model()