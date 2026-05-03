# src/train.py

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data, split_data
from src.evaluate import evaluate_regression_model, print_metrics
from src.features import build_preprocessing_pipeline


RANDOM_STATE = 42
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"


def train_model():
    """
    Entrena un modelo de regresión para predecir el valor mediano de viviendas
    usando el dataset Boston Housing.
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

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment("housing-price-regression")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regression_model(y_test, y_pred)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mape", metrics["mape"])
        mlflow.log_metric("r2", metrics["r2"])
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print("Modelo entrenado correctamente")
        print(f"Modelo guardado en: {MODEL_PATH}")
        print_metrics(metrics)


if __name__ == "__main__":
    train_model()