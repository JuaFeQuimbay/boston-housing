# 🏠 Housing Price Prediction API (MLOps Project)

## 📌 Overview

This project implements an end-to-end MLOps pipeline to train, evaluate,
and serve a machine learning model for predicting housing prices based
on the Boston Housing dataset.

The solution is fully **portable and cloud-agnostic**, using open-source
tools and designed following best practices in MLOps.

---

## ⚙️ Features

- Reproducible ML pipeline (training, evaluation, persistence)
- REST API for real-time inference (FastAPI + uvicorn / uvloop)
- Modular project structure
- Unit testing with pytest
- CI pipeline with GitHub Actions
- Multi-stage Docker image, non-root user, healthcheck
- Explainability and bias subgroup analysis

---

## 🧱 Project Structure

```
mlops-housing-api/
├── app/              # FastAPI application (production)
├── src/              # ML pipeline (training, features, evaluation, explain)
├── models/           # Trained model artifacts (model.pkl)
├── tests/            # Unit and integration tests
├── notebooks/        # EDA and explainability narrative
├── reports/          # Generated explainability and bias reports
├── docs/             # Technical test brief
├── .github/          # CI/CD workflows
├── Dockerfile        # Multi-stage image for serving
├── pyproject.toml    # Dependencies and tooling config
└── README.md
```

---

## 🚀 Getting Started

### Create environment

```powershell
python -m venv ambiente_housing_api
.\ambiente_housing_api\Scripts\activate     # Windows
# source ambiente_housing_api/bin/activate  # Linux/Mac
```

### Install dependencies

Install the dependency group you need:

```powershell
pip install ".[inference]"   # Solo servir el modelo
pip install ".[train]"       # Entrenar y experimentar
pip install ".[dev]"         # Tests
pip install ".[all]"         # Todo (recomendado en local y CI)
```

---

## 🧠 Train the model

```powershell
python -m src.train
```

This creates `models/model.pkl` plus a fresh MLflow run under `mlruns/`.

---

## 🔍 Explainability and bias analysis

```powershell
python -m src.explain
```

Generates under `reports/`:

- `bias_report.md` — auditoría de sesgo por subgrupos (`LSTAT`, `B`, `CRIM`)
- `bias_subgroup_metrics.csv` — métricas tabulares
- `feature_importance.csv`, `permutation_importance.csv`
- `figures/` — gráficos PNG

---

## 🌐 Run the API locally

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker

```powershell
docker build -t housing-api:latest .
docker run -d --name housing-api -p 8000:8000 -e MODEL_VERSION=v1 housing-api:latest

# Verify
curl http://127.0.0.1:8000/health
```

Stop and remove:

```powershell
docker stop housing-api
docker rm housing-api
```

---

## 🧪 Run tests

```powershell
pytest tests/
```

---

## 📦 Model artifact versioning (DVC)

The trained model (`models/model.pkl`, ~7 MB) is tracked with
[DVC](https://dvc.org/) instead of git. Git stores only `model.pkl.dvc`
(a 97 B pointer file with the MD5 hash); the binary lives in the DVC
cache and a remote.

### Setup

```powershell
pip install ".[mlops]"
```

### Default remote (local)

The repo is configured with a local remote at `.dvc-storage/`
(gitignored). To populate it with the current model:

```powershell
dvc push
```

To recover the binary on a fresh clone (after it has been pushed):

```powershell
dvc pull
```

### Using a real remote

For a multi-machine setup, swap the local remote for any DVC-supported
backend (S3, GCS, Azure Blob, SSH, MinIO, …) without changing the rest
of the workflow:

```powershell
dvc remote add -d origin s3://my-bucket/path
dvc push
```

### Why DVC here

- `models/model.pkl` is binary (~7 MB) and changes every retrain →
  unsuitable for git.
- The `.dvc` pointer file is a small, diffable YAML that lets git
  track which model version belongs to each commit.
- Anyone who clones the repo can recover the exact model used at any
  past commit with `git checkout <sha> && dvc pull`.

---

## 🔁 CI Pipeline

GitHub Actions automatically:
- Installs dependencies via `pip install ".[all]"`
- Trains the model
- Runs tests
- Validates the FastAPI app loads
- Validates the Docker build

---

## 📊 Model Performance

- MAE: ~2.04
- RMSE: ~2.91
- MAPE: ~11.16%
- R²: ~0.88

Disparate error ratio (worst/best subgroup MAE):
- `LSTAT`: 1.20
- `B`: 1.61
- `CRIM`: 1.50

See [reports/bias_report.md](reports/bias_report.md) for full analysis.

---

## ⚠️ Notes

- The Boston Housing dataset is deprecated in scikit-learn 1.2 due to
  the racially-biased construction of the `B` feature.
  This project uses it for academic and MLOps demonstration purposes
  only; not suitable for real-world property valuation.
- The bias report flags this explicitly and proposes mitigation paths.

---

## 🤖 AI tooling disclosure

Code structure, refactoring decisions, Dockerfile optimization and
explainability/bias analysis were assisted by Claude (Anthropic).
All design decisions were validated against the technical brief.

---

## 👤 Author

Juan Felipe Quimbay
