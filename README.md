# 🚀 MLOps Drift Detection System

A production-grade MLOps pipeline with **real-time predictions**, **automated drift detection**, **self-healing retraining**, and **model versioning** — built in 10 days.

[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github)](/.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Training │───▶│  Model   │───▶│   API    │───▶│Dashboard │ │
│  │ Pipeline │    │ Registry │    │ (FastAPI)│    │(Streamlit)│ │
│  │ + MLflow │    │ v1,v2... │    │ :8000    │    │ :8501    │ │
│  └────▲─────┘    └──────────┘    └────┬─────┘    └──────────┘ │
│       │                               │                        │
│       │         ┌──────────┐          │                        │
│       └─────────│  Drift   │◀─────────┘                        │
│   Auto-Retrain  │Detection │   Prediction                      │
│                 │ KS + PSI │   Logging                         │
│                 └──────────┘                                   │
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │  Docker  │    │  CI/CD   │    │  SQLite  │                 │
│  │ Compose  │    │ Actions  │    │ Logging  │                 │
│  └──────────┘    └──────────┘    └──────────┘                 │
└────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🔮 **Prediction API** | FastAPI server with `/predict`, `/health`, `/reload` endpoints |
| 📊 **Monitoring Dashboard** | Streamlit UI: metrics, charts, prediction history |
| 🔍 **Drift Detection** | KS-test + PSI detects data distribution shifts |
| 🔄 **Auto-Retraining** | Self-healing: drift → retrain → promote → reload |
| 📦 **Model Registry** | Versioned models (v1, v2...) + production promotion |
| 🐳 **Docker** | One-command deployment with `docker compose up` |
| ✅ **CI/CD** | GitHub Actions: pytest → Docker build on push |
| 📈 **MLflow Tracking** | Experiment tracking with params, metrics, artifacts |

## 📸 Screenshots

### FastAPI Documentation (Swagger UI)
![API Docs](assets/api_docs.png)

### Streamlit Monitoring Dashboard
![Dashboard](assets/dashboard.png)

## 📁 Project Structure

```
drift_detection_MLOPs/
├── api/                    # FastAPI inference server
│   ├── main.py             #   App + routes
│   ├── models.py           #   Pydantic schemas
│   ├── predictor.py        #   Model loading + inference
│   └── database.py         #   SQLite prediction logging
├── training/               # ML training pipeline
│   ├── config.py           #   Hyperparameters + paths
│   ├── preprocess.py       #   Data loading + scaling
│   ├── train.py            #   Training + MLflow tracking
│   ├── evaluate.py         #   Metrics calculation
│   └── retrain_pipeline.py #   Auto-retrain orchestrator
├── drift/                  # Drift detection
│   ├── drift_check.py      #   KS-test + PSI analysis
│   └── simulate_drift.py   #   Generate drifted data
├── dashboard/              # Streamlit monitoring UI
│   └── app.py              #   Dashboard with charts
├── registry/               # Model management
│   └── promote_model.py    #   Promote best → production
├── models/                 # Trained models
│   ├── v1/, v2/, v3/       #   Versioned models
│   └── production/         #   Current production model
├── data/                   # Datasets + prediction DB
├── tests/                  # pytest test suite (16 tests)
├── docker/                 # Dockerfiles
├── .github/workflows/      # CI/CD pipeline
├── docker-compose.yml      # Multi-service deployment
└── requirements.txt        # Python dependencies
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn (RandomForestRegressor) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Drift Detection | scipy (KS-test) + custom PSI |
| Database | SQLite |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest (16 tests) |

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone
git clone <repo-url>
cd drift_detection_MLOPs

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train & promote model
python -m training.train
python -m registry.promote_model

# Start API (terminal 1)
uvicorn api.main:app --reload

# Start dashboard (terminal 2)
streamlit run dashboard/app.py
```

### Option 2: Docker

```bash
docker compose build
docker compose up
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Make a prediction |
| `GET` | `/predictions` | View prediction history |
| `POST` | `/reload` | Hot-reload production model |

### Example: Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 322,
    "AveOccup": 2.5,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Response:
```json
{
  "prediction": 4.31,
  "model_version": 1,
  "timestamp": "2026-02-08T23:03:59",
  "prediction_id": 1
}
```

## 🔍 Drift Detection

```bash
# Check drift with your data
python -m drift.drift_check --file data/incoming.csv

# Simulate drifted data for testing
python -m drift.simulate_drift

# Run auto-retraining pipeline
python -m training.retrain_pipeline --simulate
```

### Drift Detection Methods

| Method | Threshold | Interpretation |
|--------|-----------|----------------|
| **KS-test** | p < 0.05 | Significant distribution shift |
| **PSI** | > 0.2 | Population stability changed |

Drift triggers retraining when **≥25%** of features show drift.

## 🔄 Auto-Retraining Pipeline

The self-healing pipeline runs automatically:

```
📥 Load incoming data
   ↓
🔍 Check for drift (KS + PSI)
   ↓
🚀 Retrain model (if drift detected)
   ↓
📊 Evaluate new model
   ↓
🏭 Promote to production (if acceptable)
   ↓
🔄 Reload API model
```

```bash
# With simulated drift
python -m training.retrain_pipeline --simulate

# With real data
python -m training.retrain_pipeline --data path/to/data.csv

# Force retrain (skip drift check)
python -m training.retrain_pipeline --force
```

## ✅ Testing

```bash
# Run all tests
pytest tests/ -v

# 16 tests across 3 modules:
# test_api.py      — 4 tests (health, predict, validation)
# test_drift.py    — 6 tests (PSI, KS-test, drift scenarios)
# test_training.py — 6 tests (data, training, evaluation)
```

## 📊 Dataset

**California Housing** (scikit-learn built-in)
- 20,640 samples, 8 features
- Regression: predicting median house value ($100K units)
- Features: median income, house age, avg rooms, population, location, etc.

## 📈 10-Day Build Progress

- [x] **Day 1** — Project skeleton + dataset
- [x] **Day 2** — Training pipeline (R² = 0.775)
- [x] **Day 3** — MLflow + model versioning
- [x] **Day 4** — FastAPI inference server
- [x] **Day 5** — Streamlit monitoring dashboard
- [x] **Day 6** — Drift detection (KS-test + PSI)
- [x] **Day 7** — Auto-retraining loop
- [x] **Day 8** — Docker containerization
- [x] **Day 9** — CI/CD with GitHub Actions
- [x] **Day 10** — Deployment + README

## 📄 License

MIT
