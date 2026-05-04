# src/explain.py

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.data import load_data, split_data


RANDOM_STATE = 42
N_QUANTILES = 3
QUANTILE_LABELS = ["Low", "Medium", "High"]

MODEL_PATH = Path("models/model.pkl")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
PERMUTATION_IMPORTANCE_PATH = REPORTS_DIR / "permutation_importance.csv"
BIAS_REPORT_PATH = REPORTS_DIR / "bias_report.md"
BIAS_TABLE_PATH = REPORTS_DIR / "bias_subgroup_metrics.csv"

# Variables sensibles a auditar (proxies socioeconómicos del dataset)
SENSITIVE_FEATURES = ["LSTAT", "B", "CRIM"]


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero: python -m src.train"
        )

    return joblib.load(MODEL_PATH)


def compute_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Importancia nativa del modelo Random Forest."""

    rf_model = model.named_steps["model"]

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:
    """Permutation importance sobre el set de test."""

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    top_features = importance_df.head(10).sort_values(
        by=importance_df.columns[1],
        ascending=True,
    )

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features.iloc[:, 1])
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_partial_dependence(model, X: pd.DataFrame, features: List[str], output_path: Path) -> None:
    _, ax = plt.subplots(figsize=(10, 6))

    PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_subgroup_metrics(
    df: pd.DataFrame,
    feature: str,
    y_col: str = "MEDV",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """
    Calcula métricas de error por terciles de una variable sensible.

    Devuelve por subgrupo: n, mean(y), mean(pred), MAE, MAPE, bias firmado.
    """

    df = df.copy()
    df["__group__"] = pd.qcut(
        df[feature],
        q=N_QUANTILES,
        labels=QUANTILE_LABELS,
        duplicates="drop",
    )

    rows = []
    for group, sub in df.groupby("__group__", observed=True):
        y_true = sub[y_col].to_numpy()
        y_pred = sub[pred_col].to_numpy()

        mae = mean_absolute_error(y_true, y_pred)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        signed_bias = float(np.mean(y_pred - y_true))

        rows.append(
            {
                "feature": feature,
                "subgroup": str(group),
                "n": int(len(sub)),
                "mean_actual": float(np.mean(y_true)),
                "mean_pred": float(np.mean(y_pred)),
                "mae": float(mae),
                "mape": mape,
                "signed_bias": signed_bias,
            }
        )

    return pd.DataFrame(rows)


def disparate_error_ratio(subgroup_df: pd.DataFrame, metric: str = "mae") -> float:
    """Ratio entre el peor y el mejor subgrupo. 1.0 = paridad perfecta."""

    values = subgroup_df[metric]
    return float(values.max() / values.min()) if values.min() > 0 else float("inf")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a tabla markdown sin depender de `tabulate`."""

    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, separator, *rows])


def render_bias_report(
    overall_metrics: Dict[str, float],
    subgroup_tables: Dict[str, pd.DataFrame],
    feature_importance: pd.DataFrame,
    permutation_imp: pd.DataFrame,
) -> str:
    """Genera el contenido markdown del reporte de sesgo."""

    lines: List[str] = []
    lines.append("# Reporte de Explicabilidad y Sesgo del Modelo")
    lines.append("")
    lines.append("Modelo: `RandomForestRegressor` sobre Boston Housing.")
    lines.append("Generado automáticamente por `src/explain.py`.")
    lines.append("")

    lines.append("## 1. Métricas globales (test set)")
    lines.append("")
    lines.append(f"- **MAE**: {overall_metrics['mae']:.4f}")
    lines.append(f"- **MAPE**: {overall_metrics['mape']:.2f}%")
    lines.append(f"- **R²**: {overall_metrics['r2']:.4f}")
    lines.append("")

    lines.append("## 2. Variables más influyentes")
    lines.append("")
    lines.append("Top 5 por importancia nativa (Gini):")
    lines.append("")
    lines.append(_df_to_markdown(feature_importance.head(5)))
    lines.append("")
    lines.append("Top 5 por permutation importance (más robusto):")
    lines.append("")
    lines.append(_df_to_markdown(permutation_imp.head(5)))
    lines.append("")

    lines.append("## 3. Análisis de sesgo por subgrupos")
    lines.append("")
    lines.append(
        "Se evalúa el error por terciles de variables socioeconómicas. "
        "Un modelo equitativo debería tener errores similares entre subgrupos."
    )
    lines.append("")

    for feature, table in subgroup_tables.items():
        ratio = disparate_error_ratio(table, "mae")
        lines.append(f"### {feature}")
        lines.append("")
        lines.append(_df_to_markdown(table.round(4)))
        lines.append("")
        lines.append(f"**Disparate error ratio (max MAE / min MAE)**: {ratio:.2f}")
        lines.append("")

        signed = table["signed_bias"].to_numpy()
        if (signed > 0).any() and (signed < 0).any():
            lines.append(
                "Se observa **bias firmado opuesto entre subgrupos**: el modelo "
                "sobreestima en algunos terciles y subestima en otros."
            )
        else:
            lines.append(
                "El bias firmado tiene el mismo signo en todos los subgrupos: "
                "no hay reversión de error, pero la magnitud puede variar."
            )
        lines.append("")

    lines.append("## 4. Consideraciones éticas")
    lines.append("")
    lines.append(
        "El dataset Boston Housing fue **deprecado en scikit-learn 1.2** debido a la "
        "variable `B = 1000(Bk - 0.63)²`, donde `Bk` es la proporción de población "
        "afroamericana por barrio. Su construcción asume que existe un valor "
        "\"ideal\" de composición racial, lo cual codifica un sesgo discriminatorio "
        "explícito."
    )
    lines.append("")
    lines.append(
        "En este proyecto se mantiene `B` únicamente con fines académicos para "
        "ilustrar capacidades de auditoría y MLOps; **no debe usarse en un sistema "
        "real de valuación inmobiliaria**. Recomendaciones para producción:"
    )
    lines.append("")
    lines.append("- Eliminar `B` del set de features y reentrenar.")
    lines.append("- Reemplazar el dataset por uno actual (p. ej. California Housing).")
    lines.append(
        "- Implementar monitoreo continuo de fairness en producción "
        "(disparate error, demographic parity sobre proxies socioeconómicos)."
    )
    lines.append(
        "- Documentar la población objetivo y los límites de uso del modelo "
        "(model card)."
    )
    lines.append("")

    lines.append("## 5. Artefactos generados")
    lines.append("")
    lines.append("- `reports/feature_importance.csv`")
    lines.append("- `reports/permutation_importance.csv`")
    lines.append("- `reports/bias_subgroup_metrics.csv`")
    lines.append("- `reports/figures/feature_importance.png`")
    lines.append("- `reports/figures/permutation_importance.png`")
    lines.append("- `reports/figures/partial_dependence_rm_lstat.png`")
    lines.append("")

    return "\n".join(lines)


def explain_model() -> None:
    """Ejecuta el análisis de explicabilidad y sesgo."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = split_data(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = load_trained_model()
    feature_names = X.columns.tolist()

    feature_importance_df = compute_feature_importance(model, feature_names)
    permutation_importance_df = compute_permutation_importance(
        model, X_test, y_test, feature_names
    )

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    permutation_importance_df.to_csv(PERMUTATION_IMPORTANCE_PATH, index=False)

    plot_feature_importance(
        feature_importance_df,
        FIGURES_DIR / "feature_importance.png",
    )
    plot_feature_importance(
        permutation_importance_df.rename(columns={"importance_mean": "importance"}),
        FIGURES_DIR / "permutation_importance.png",
    )
    plot_partial_dependence(
        model,
        X_test,
        ["RM", "LSTAT"],
        FIGURES_DIR / "partial_dependence_rm_lstat.png",
    )

    # Métricas globales y por subgrupo (solo sobre test para evitar leakage)
    y_pred_test = model.predict(X_test)
    test_df = X_test.copy()
    test_df["MEDV"] = y_test.to_numpy()
    test_df["prediction"] = y_pred_test

    mae = float(mean_absolute_error(y_test, y_pred_test))
    mape = float(np.mean(np.abs((y_test.to_numpy() - y_pred_test) / y_test.to_numpy())) * 100)
    ss_res = float(np.sum((y_test.to_numpy() - y_pred_test) ** 2))
    ss_tot = float(np.sum((y_test.to_numpy() - np.mean(y_test.to_numpy())) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    overall = {"mae": mae, "mape": mape, "r2": r2}

    subgroup_tables: Dict[str, pd.DataFrame] = {}
    all_subgroups: List[pd.DataFrame] = []
    for feature in SENSITIVE_FEATURES:
        table = compute_subgroup_metrics(test_df, feature)
        subgroup_tables[feature] = table
        all_subgroups.append(table)

    pd.concat(all_subgroups, ignore_index=True).to_csv(BIAS_TABLE_PATH, index=False)

    report_md = render_bias_report(
        overall_metrics=overall,
        subgroup_tables=subgroup_tables,
        feature_importance=feature_importance_df,
        permutation_imp=permutation_importance_df,
    )
    BIAS_REPORT_PATH.write_text(report_md, encoding="utf-8")

    print("Explicabilidad y análisis de sesgo generados correctamente")
    print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"Permutation importance: {PERMUTATION_IMPORTANCE_PATH}")
    print(f"Bias subgroup metrics: {BIAS_TABLE_PATH}")
    print(f"Reporte de sesgo: {BIAS_REPORT_PATH}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    explain_model()
