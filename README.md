# 🏠 Housing Price Prediction API (MLOps Project)

🌍 [Leer en español](README.es.md)

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
- Model artifact versioning with DVC

---

## 🧱 Project Structure

```
mlops-housing-api/
├── app/              # FastAPI application (production)
├── src/              # ML pipeline (training, features, evaluation, explain)
├── models/           # Trained model artifacts (model.pkl)
├── tests/            # Unit and integration tests
├── notebooks/        # EDA narrative
├── reports/          # Generated explainability and bias reports
├── docs/             # Technical brief (gitignored)
├── .github/          # CI/CD workflows
├── .dvc/             # DVC config
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
pip install ".[inference]"   # Serve only
pip install ".[train]"       # Training and experimentation
pip install ".[dev]"         # Tests
pip install ".[mlops]"       # DVC
pip install ".[all]"         # Everything (recommended in local and CI)
```

---

## 🧠 Train the model

```powershell
python -m src.train
```

This creates `models/model.pkl`, `models/model_metadata.json`, plus a
fresh MLflow run under `mlruns/`.

---

## 🔍 Explainability and bias analysis

```powershell
python -m src.explain
```

Generates under `reports/`:

- `bias_report.md` — bias audit by subgroup (`LSTAT`, `B`, `CRIM`)
- `bias_subgroup_metrics.csv` — tabular metrics
- `feature_importance.csv`, `permutation_importance.csv`
- `figures/` — PNG plots

---

## 🌐 Run the API locally

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker

Build the image:

```powershell
docker build -t housing-api:latest .
```

Run the container:

```powershell
docker run -d --name housing-api -p 8000:8000 housing-api:latest
```

### Example request

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

Response:

```json
{
  "prediction": 28.5845,
  "model_version": "de2171be"
}
```

Health check (returns model version, training timestamp, git SHA):

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "de2171be",
  "trained_at": "2026-05-04T03:47:56.744390+00:00",
  "git_sha": "200155e"
}
```

Stop and remove:

```powershell
docker stop housing-api ; docker rm housing-api
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

## 🚀 Possible Improvements

The current solution is a working MVP. Natural next steps grouped
by maturity:

### Modeling
- Replace Boston Housing with a non-deprecated dataset (e.g. California
  Housing) to remove the ethically problematic `B` feature.
- Try gradient-boosting models (LightGBM / XGBoost) — typically 3× faster
  inference for similar R² on tabular data.
- Tune `n_estimators` and `max_depth` of the Random Forest: the current
  model uses 200 estimators with unlimited depth, which dominates the
  ~31 ms request latency.

### MLOps
- Replace the local DVC remote with S3 / GCS / MinIO for multi-machine
  reproducibility.
- Run an MLflow Tracking Server (Postgres + artifact store) instead of
  filesystem, so experiments are visible across the team.
- Promote models via MLflow Model Registry stages
  (`Staging → Production → Archived`) with automated gating on
  `R²` / `disparate_error_ratio` thresholds.
- Define a `dvc.yaml` pipeline (`prepare → train → explain`) so
  `dvc repro` re-executes only what changed.

### Serving
- Add Prometheus metrics endpoint and a Grafana dashboard for latency
  P50/P95, request rate and prediction distribution drift.
- Harden input validation against out-of-domain ranges (currently
  Pydantic only validates types).
- Multi-worker uvicorn behind a reverse proxy (nginx / Traefik) for
  horizontal scaling.

### Quality / Governance
- Add data drift monitoring (Evidently AI or similar).
- Add automated bias regression tests in CI: fail the build if
  disparate error ratio on `LSTAT`/`B`/`CRIM` exceeds a threshold.
- Add a model card under `docs/` with intended use, limitations and
  performance per subgroup.

### Security
- Sign Docker images (cosign / Sigstore) and verify in the deployment
  pipeline.
- Add API key / OAuth on the `/predict` endpoint for production use.
- Scan dependencies on each PR (Dependabot / pip-audit).

---

## ⚠️ Notes

- The Boston Housing dataset is deprecated in scikit-learn 1.2 due to
  the racially-biased construction of the `B` feature.
  This project uses it for academic and MLOps demonstration purposes
  only; not suitable for real-world property valuation.
- The bias report flags this explicitly and proposes mitigation paths.

---

## 🤖 AI tooling disclosure

This project was developed with assistance from two AI tools, used
for distinct purposes:

### Claude (Anthropic) — paired programming

Through Claude Code in VS Code, used as a paired-programming
collaborator for:

- Refactoring the original Dockerfile into a multi-stage build with
  non-root user, healthcheck and `uvloop`/`httptools` for latency.
- Designing the bias subgroup analysis in `src/explain.py` (LSTAT, B,
  CRIM) and the auto-generated `bias_report.md`.
- Migrating dependency management from `requirements.txt` to
  `pyproject.toml` with optional dependency groups.
- Introducing DVC for model artifact versioning with a local remote.
- Writing the model metadata pipeline (`model_metadata.json` →
  `/health` endpoint exposing `model_version`, `trained_at`, `git_sha`).

### Perplexity — Deep Search on the dataset

Perplexity Deep Search was used to investigate the Boston Housing
dataset itself: its origin, documented issues with the `B` feature
(racially-biased construction), and the reasons behind its deprecation
in scikit-learn 1.2. Findings were validated against primary sources
before being incorporated into [reports/bias_report.md](reports/bias_report.md)
and the ethical considerations of this README.

### Authorship and validation

Architectural decisions, tradeoffs (e.g. choosing a local DVC remote
over a cloud-managed one to remain cloud-agnostic per the brief),
the choice of Random Forest over alternatives, and all final reviews
were made by the author. Every AI suggestion was validated by reading
diffs, running tests and benchmarking before committing.

---

## 👤 Author

Juan Felipe Quimbay
