# 🏠 API de Predicción de Precios de Viviendas (Proyecto MLOps)

🌍 [Read in English](README.md)

## 📌 Visión general

Este proyecto implementa un pipeline de MLOps end-to-end para entrenar,
evaluar y servir un modelo de Machine Learning que predice precios de
viviendas utilizando el dataset Boston Housing.

La solución es **portable y agnóstica de la nube**, basada
exclusivamente en herramientas open-source y diseñada siguiendo
buenas prácticas de MLOps.

---

## ⚙️ Características

- Pipeline de ML reproducible (entrenamiento, evaluación, persistencia)
- API REST para inferencia en tiempo real (FastAPI + uvicorn / uvloop)
- Estructura de proyecto modular
- Tests unitarios con pytest
- CI con GitHub Actions
- Imagen Docker multi-stage, usuario no-root, healthcheck
- Análisis de explicabilidad y sesgo por subgrupos
- Versionado de artefactos del modelo con DVC

---

## 🧱 Estructura del proyecto

```
mlops-housing-api/
├── app/              # Aplicación FastAPI (producción)
├── src/              # Pipeline de ML (training, features, evaluación, explain)
├── models/           # Artefactos del modelo entrenado
├── tests/            # Tests unitarios y de integración
├── notebooks/        # EDA narrativo
├── reports/          # Reportes de explicabilidad y sesgo
├── docs/             # Brief técnico (gitignored)
├── .github/          # Workflows de CI/CD
├── .dvc/             # Configuración de DVC
├── Dockerfile        # Imagen multi-stage para servir
├── pyproject.toml    # Dependencias y configuración de tooling
└── README.md
```

---

## 🚀 Cómo arrancar

### Crear entorno virtual

```powershell
python -m venv ambiente_housing_api
.\ambiente_housing_api\Scripts\activate     # Windows
# source ambiente_housing_api/bin/activate  # Linux/Mac
```

### Instalar dependencias

Instala el grupo que necesites:

```powershell
pip install ".[inference]"   # Solo servir el modelo
pip install ".[train]"       # Entrenar y experimentar
pip install ".[dev]"         # Tests
pip install ".[mlops]"       # DVC
pip install ".[all]"         # Todo (recomendado en local y CI)
```

---

## 🧠 Entrenar el modelo

```powershell
python -m src.train
```

Genera `models/model.pkl`, `models/model_metadata.json` y un nuevo run
en MLflow bajo `mlruns/`.

---

## 🔍 Análisis de explicabilidad y sesgo

```powershell
python -m src.explain
```

Genera bajo `reports/`:

- `bias_report.md` — auditoría de sesgo por subgrupos (`LSTAT`, `B`, `CRIM`)
- `bias_subgroup_metrics.csv` — métricas tabulares
- `feature_importance.csv`, `permutation_importance.csv`
- `figures/` — gráficos PNG

---

## 🌐 Ejecutar la API en local

```powershell
uvicorn app.main:app --reload
```

Abre http://127.0.0.1:8000/docs

---

## 🐳 Ejecutar con Docker

Construir la imagen:

```powershell
docker build -t housing-api:latest .
```

Levantar el contenedor:

```powershell
docker run -d --name housing-api -p 8000:8000 housing-api:latest
```

### Ejemplo de request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0,
        "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
        "RAD": 1, "TAX": 296.0, "PTRATIO": 15.3,
        "B": 396.9, "LSTAT": 4.98
      }'
```

Respuesta:

```json
{
  "prediction": 28.5845,
  "model_version": "de2171be"
}
```

Healthcheck (devuelve la versión del modelo, timestamp de
entrenamiento y git SHA):

```bash
curl http://127.0.0.1:8000/health
```

Respuesta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "de2171be",
  "trained_at": "2026-05-04T03:47:56.744390+00:00",
  "git_sha": "200155e"
}
```

Detener y eliminar:

```powershell
docker stop housing-api ; docker rm housing-api
```

---

## 🧪 Ejecutar tests

```powershell
pytest tests/
```

---

## 📦 Versionado de artefactos del modelo (DVC)

El modelo entrenado (`models/model.pkl`, ~7 MB) se versiona con
[DVC](https://dvc.org/) en lugar de git. Git solo guarda
`model.pkl.dvc` (un archivo de 97 B con el hash MD5); el binario vive
en el caché de DVC y en un remote.

### Setup

```powershell
pip install ".[mlops]"
```

### Remote por defecto (local)

El repositorio está configurado con un remote local en `.dvc-storage/`
(gitignored). Para poblarlo con el modelo actual:

```powershell
dvc push
```

Para recuperar el binario tras un clone limpio (después de haber hecho
push):

```powershell
dvc pull
```

### Usar un remote real

Para un setup multi-máquina, basta con cambiar el remote local por
cualquier backend que soporte DVC (S3, GCS, Azure Blob, SSH, MinIO, …)
sin tocar nada más:

```powershell
dvc remote add -d origin s3://mi-bucket/path
dvc push
```

### Por qué DVC aquí

- `models/model.pkl` es binario (~7 MB) y cambia con cada
  reentrenamiento → no apto para git.
- El archivo `.dvc` es un YAML pequeño y *diffable* que permite a git
  saber qué versión del modelo corresponde a cada commit.
- Cualquiera que clone el repo puede recuperar el modelo exacto de
  cualquier commit pasado con `git checkout <sha> && dvc pull`.

---

## 🔁 Pipeline de CI

GitHub Actions automatiza:
- Instalación de dependencias con `pip install ".[all]"`
- Entrenamiento del modelo
- Ejecución de tests
- Validación de carga de la app FastAPI
- Validación del build de Docker

---

## 📊 Performance del modelo

- MAE: ~2.04
- RMSE: ~2.91
- MAPE: ~11.16%
- R²: ~0.88

Disparate error ratio (peor/mejor MAE por subgrupo):
- `LSTAT`: 1.20
- `B`: 1.61
- `CRIM`: 1.50

Análisis completo en [reports/bias_report.md](reports/bias_report.md).

---

## 🚀 Posibles mejoras

La solución actual es un MVP funcional. Próximos pasos naturales
agrupados por madurez:

### Modelado
- Reemplazar Boston Housing por un dataset no deprecado (p. ej.
  California Housing) para eliminar la variable `B`, problemática
  éticamente.
- Probar modelos de gradient boosting (LightGBM / XGBoost) — suelen ser
  ~3× más rápidos en inferencia con R² similar para datos tabulares.
- Ajustar `n_estimators` y `max_depth` del Random Forest: el modelo
  actual usa 200 estimadores sin límite de profundidad, lo que domina
  los ~31 ms de latencia por request.

### MLOps
- Reemplazar el remote local de DVC por S3 / GCS / MinIO para
  reproducibilidad multi-máquina.
- Levantar un MLflow Tracking Server (Postgres + artifact store) en
  vez del backend de filesystem, para que los experimentos sean
  visibles para todo el equipo.
- Promover modelos vía MLflow Model Registry con stages
  (`Staging → Production → Archived`) y gates automáticos sobre
  umbrales de `R²` / `disparate_error_ratio`.
- Definir un pipeline `dvc.yaml` (`prepare → train → explain`) para
  que `dvc repro` re-ejecute solo lo que cambió.

### Serving
- Endpoint de métricas Prometheus y dashboard en Grafana para latencia
  P50/P95, request rate y drift de la distribución de predicciones.
- Endurecer la validación de entrada contra rangos fuera de dominio
  (Pydantic hoy solo valida tipos).
- Uvicorn multi-worker detrás de un reverse proxy (nginx / Traefik)
  para escalar horizontalmente.

### Calidad / Gobierno
- Monitoreo de data drift (Evidently AI u otra similar).
- Tests automáticos de regresión de sesgo en CI: fallar el build si el
  disparate error ratio en `LSTAT`/`B`/`CRIM` supera un umbral.
- Model card bajo `docs/` con uso previsto, limitaciones y performance
  por subgrupo.

### Seguridad
- Firmar imágenes Docker (cosign / Sigstore) y verificarlas en el
  pipeline de deployment.
- Añadir API key / OAuth al endpoint `/predict` para producción.
- Escaneo de dependencias en cada PR (Dependabot / pip-audit).

---

## ⚠️ Notas

- El dataset Boston Housing fue deprecado en scikit-learn 1.2 debido a
  la construcción racialmente sesgada de la variable `B`. Este proyecto
  lo utiliza con fines académicos y de demostración de MLOps
  únicamente; **no debe usarse en valuación inmobiliaria real**.
- El reporte de sesgo lo señala explícitamente y propone caminos de
  mitigación.

---

## 🤖 Uso de herramientas de IA

Este proyecto se desarrolló con asistencia de dos herramientas de IA,
usadas con propósitos distintos:

### Claude (Anthropic) — paired programming

A través de Claude Code en VS Code, en modo paired-programming para:

- Refactorizar el Dockerfile original a un build multi-stage con
  usuario no-root, healthcheck y `uvloop` / `httptools` para latencia.
- Diseñar el análisis de sesgo por subgrupos en `src/explain.py`
  (LSTAT, B, CRIM) y el `bias_report.md` autogenerado.
- Migrar la gestión de dependencias de `requirements.txt` a
  `pyproject.toml` con grupos opcionales.
- Introducir DVC para versionar el artefacto del modelo con un remote
  local.
- Implementar el pipeline de metadata del modelo
  (`model_metadata.json` → endpoint `/health` con `model_version`,
  `trained_at` y `git_sha`).

### Perplexity — Deep Search sobre el dataset

Se utilizó Perplexity Deep Search para investigar el dataset Boston
Housing: su origen, los problemas documentados de la variable `B`
(construcción racialmente sesgada) y las razones detrás de su
deprecación en scikit-learn 1.2. Los hallazgos fueron validados
contra fuentes primarias antes de incorporarse a
[reports/bias_report.md](reports/bias_report.md) y a las
consideraciones éticas de este README.

### Autoría y validación

Las decisiones arquitecturales, los tradeoffs (p. ej. usar un remote
local de DVC en lugar de uno gestionado en la nube para mantenerse
cloud-agnostic según el brief), la elección de Random Forest sobre
alternativas, y todas las revisiones finales fueron del autor. Cada
sugerencia de la IA fue validada leyendo diffs, ejecutando tests y
benchmarks antes de commit.

---

## 👤 Autor

Juan Felipe Quimbay
