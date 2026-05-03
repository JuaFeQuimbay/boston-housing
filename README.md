# 🏠 Housing Price Prediction API (MLOps Project)

## 📌 Overview

This project implements an end-to-end MLOps pipeline to train, evaluate,
and serve a machine learning model for predicting housing prices based
on the Boston Housing dataset.

The solution is fully **portable and cloud-agnostic**, using open-source
tools and designed following best practices in MLOps.

------------------------------------------------------------------------

## ⚙️ Features

-   Reproducible ML pipeline (training, evaluation, persistence)
-   REST API for real-time inference (FastAPI)
-   Modular project structure
-   Unit testing with pytest
-   CI pipeline with GitHub Actions
-   Docker-ready for deployment

------------------------------------------------------------------------

## 🧱 Project Structure

mlops-housing-api/ ├── app/ \# FastAPI application ├── src/ \# ML
pipeline ├── models/ \# Trained model artifacts ├── tests/ \# Unit tests
├── notebooks/ \# EDA ├── .github/ \# CI/CD pipeline ├── Dockerfile ├──
docker-compose.yml ├── requirements.txt └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### Create environment

python -m venv ambiente_housing_api

Activate: Windows:
.`\ambiente`{=tex}\_housing_api`\Scripts`{=tex}`\activate`{=tex}

Linux/Mac: source ambiente_housing_api/bin/activate

------------------------------------------------------------------------

### Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

## 🧠 Train the model

python -m src.train

------------------------------------------------------------------------

## 🌐 Run the API

uvicorn app.main:app --reload

Go to: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## 🧪 Run Tests

pytest tests/

------------------------------------------------------------------------

## 🔁 CI Pipeline

GitHub Actions automatically: - installs dependencies - trains model -
runs tests - validates API - validates Docker build

------------------------------------------------------------------------

## 📊 Model Performance

-   MAE: \~2.05
-   RMSE: \~2.91
-   MAPE: \~11.28%
-   R²: \~0.88

------------------------------------------------------------------------

## ⚠️ Notes

-   Dataset may contain socio-economic bias
-   For demonstration purposes

------------------------------------------------------------------------

## 🚀 Author

Juan Felipe Quimbay
