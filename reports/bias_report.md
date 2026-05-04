# Reporte de Explicabilidad y Sesgo del Modelo

Modelo: `RandomForestRegressor` sobre Boston Housing.
Generado automáticamente por `src/explain.py`.

## 1. Métricas globales (test set)

- **MAE**: 2.0422
- **MAPE**: 11.16%
- **R²**: 0.8839

## 2. Variables más influyentes

Top 5 por importancia nativa (Gini):

| feature | importance |
| --- | --- |
| RM | 0.4923 |
| LSTAT | 0.3235 |
| DIS | 0.0563 |
| CRIM | 0.0385 |
| NOX | 0.0160 |

Top 5 por permutation importance (más robusto):

| feature | importance_mean | importance_std |
| --- | --- | --- |
| LSTAT | 2.5028 | 0.3168 |
| RM | 2.1189 | 0.1222 |
| CRIM | 0.4212 | 0.0884 |
| DIS | 0.3409 | 0.1383 |
| NOX | 0.2617 | 0.0617 |

## 3. Análisis de sesgo por subgrupos

Se evalúa el error por terciles de variables socioeconómicas. Un modelo equitativo debería tener errores similares entre subgrupos.

### LSTAT

| feature | subgroup | n | mean_actual | mean_pred | mae | mape | signed_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTAT | Low | 34 | 28.7441 | 28.2592 | 1.8616 | 6.4522 | -0.4849 |
| LSTAT | Medium | 34 | 21.1412 | 21.2683 | 2.2261 | 10.9806 | 0.1271 |
| LSTAT | High | 34 | 14.5794 | 14.5650 | 2.0387 | 16.0404 | -0.0144 |

**Disparate error ratio (max MAE / min MAE)**: 1.20

Se observa **bias firmado opuesto entre subgrupos**: el modelo sobreestima en algunos terciles y subestima en otros.

### B

| feature | subgroup | n | mean_actual | mean_pred | mae | mape | signed_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | Low | 34 | 19.0147 | 18.2491 | 2.4771 | 15.1298 | -0.7656 |
| B | Medium | 34 | 24.4794 | 24.3480 | 1.5406 | 6.7369 | -0.1314 |
| B | High | 34 | 20.9706 | 21.4953 | 2.1088 | 11.6064 | 0.5247 |

**Disparate error ratio (max MAE / min MAE)**: 1.61

Se observa **bias firmado opuesto entre subgrupos**: el modelo sobreestima en algunos terciles y subestima en otros.

### CRIM

| feature | subgroup | n | mean_actual | mean_pred | mae | mape | signed_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRIM | Low | 34 | 26.4824 | 26.2776 | 1.7668 | 6.8179 | -0.2047 |
| CRIM | Medium | 34 | 21.0559 | 21.2780 | 1.7409 | 10.3385 | 0.2221 |
| CRIM | High | 34 | 16.9265 | 16.5369 | 2.6188 | 16.3168 | -0.3896 |

**Disparate error ratio (max MAE / min MAE)**: 1.50

Se observa **bias firmado opuesto entre subgrupos**: el modelo sobreestima en algunos terciles y subestima en otros.

## 4. Consideraciones éticas

El dataset Boston Housing fue **deprecado en scikit-learn 1.2** debido a la variable `B = 1000(Bk - 0.63)²`, donde `Bk` es la proporción de población afroamericana por barrio. Su construcción asume que existe un valor "ideal" de composición racial, lo cual codifica un sesgo discriminatorio explícito.

En este proyecto se mantiene `B` únicamente con fines académicos para ilustrar capacidades de auditoría y MLOps; **no debe usarse en un sistema real de valuación inmobiliaria**. Recomendaciones para producción:

- Eliminar `B` del set de features y reentrenar.
- Reemplazar el dataset por uno actual (p. ej. California Housing).
- Implementar monitoreo continuo de fairness en producción (disparate error, demographic parity sobre proxies socioeconómicos).
- Documentar la población objetivo y los límites de uso del modelo (model card).

## 5. Artefactos generados

- `reports/feature_importance.csv`
- `reports/permutation_importance.csv`
- `reports/bias_subgroup_metrics.csv`
- `reports/figures/feature_importance.png`
- `reports/figures/permutation_importance.png`
- `reports/figures/partial_dependence_rm_lstat.png`
