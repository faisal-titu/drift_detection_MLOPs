---
title: Drift Detection MLOps
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# MLOps Drift Detection System

An end-to-end MLOps pipeline that trains a machine learning model, serves real-time predictions via a REST API, continuously monitors for **data drift**, and automatically retrains and promotes improved models — all with a live monitoring dashboard, Docker deployment, and CI/CD.

Built on the **California Housing** dataset (scikit-learn), the system predicts median house values and demonstrates production-grade ML lifecycle management.

## Live Demo

The project is deployed on Hugging Face Spaces:

[Drift Detection MLOps - Live Demo](https://huggingface.co/spaces/faisaltitu/Drift-Detection)


[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github)](/.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)

---

## Table of Contents

- [Key Numbers](#key-numbers)
- [Architecture](#architecture)
- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [SLOs & Eval Gates](#slos--eval-gates)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
  - [1. Training a Model](#1-training-a-model)
  - [2. Starting the API Server](#2-starting-the-api-server)
  - [3. Making Predictions](#3-making-predictions)
  - [4. Monitoring Dashboard](#4-monitoring-dashboard)
  - [5. Drift Detection](#5-drift-detection)
  - [6. Auto-Retraining](#6-auto-retraining)
  - [7. Streaming Synthetic Concept Drift](#7-streaming-synthetic-concept-drift)
  - [8. Observability & Metrics](#8-observability--metrics)
  - [9. Model Rollback](#9-model-rollback)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Running Tests](#running-tests)
- [Postmortem & Lessons Learned](#postmortem--lessons-learned)
- [Screenshots](#screenshots)
- [License](#license)

---

## Key Numbers

| Metric | Value | Notes |
|---|---|---|
| **Production R2** | 0.894 | RandomForestRegressor on California Housing |
| **Production RMSE** | 0.509 ($50.9K) | Median error on test split |
| **Production MAE** | 0.317 ($31.7K) | Mean absolute error on test |
| **Prediction p95 latency** | < 15 ms | Single-sample inference (measured via `/metrics`) |
| **Prediction p99 latency** | < 25 ms | Under load with Uvicorn workers |
| **/health p95 latency** | < 5 ms | Lightweight status check |
| **SLO target** | p95 < 150 ms | `/predict` endpoint |
| **Drift detection sensitivity** | ≥ 25% features | KS-test (p < 0.05) + PSI (> 0.2) |
| **Retrain eval gate** | R2 ≥ 0.70 + improvement margin | Shadow test before promotion |
| **RMSE ceiling** | ≤ 1.0 ($100K) | Hard reject if exceeded |
| **Test suite** | 16 tests, < 7s | API, drift, training modules |
| **Docker** | 2-container compose | API + Dashboard |

---

## Architecture

![Architecture](assets/Architecture.png)

<!-- ```
┌───────────────┐     ┌────────────────────────────────────────────────────┐
│  Data Source   │     │                 Inference Path                     │
│ (CSV / Stream) │────▶│  FastAPI + Uvicorn (:8000)                        │
└───────────────┘     │  ├─ /predict  → Scaler → RF Model → Response      │
                      │  ├─ /health   → Liveness + model version           │
                      │  ├─ /metrics  → p50/p95/p99, SLO breaches         │
                      │  ├─ /rollback → Promote prev version + reload      │
                      │  └─ Middleware: LatencyMiddleware (per-req timing)  │
                      └──────────┬─────────────────────────────────────────┘
                                 │ logs every prediction
                                 ▼
                      ┌────────────────────┐
                      │  SQLite            │
                      │  predictions.db    │
                      └────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
┌──────────────────┐ ┌─────────────────┐  ┌──────────────────────┐
│ Streamlit        │ │ Drift Detection │  │ MLflow Experiment     │
│ Dashboard (:8501)│ │ KS-test + PSI   │  │ Tracking (local)      │
│ Monitor|Predict| │ │ per-feature     │  │ params, metrics,      │
│ Drift|Operations │ │ ≥25% → alert    │  │ artifacts per run     │
└──────────────────┘ └────────┬────────┘  └───────────────────────┘
                              │ drift detected?
                              ▼
                    ┌──────────────────────────┐
                    │  Auto-Retrain Pipeline   │
                    │  1. Combine ref + new    │
                    │  2. Train new RF model   │
                    │  3. Eval gates:          │
                    │     • R2 ≥ 0.70          │
                    │     • R2 > prod + margin  │
                    │     • RMSE ≤ 1.0         │
                    │  4. Promote to prod/     │
                    │  5. Hot-reload API       │
                    └──────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────────┐
                    │  Model Registry          │
                    │  models/v1/ v2/ v3/ ...  │
                    │  models/production/      │
                    │  (metadata.json +        │
                    │   model.joblib +         │
                    │   scaler.joblib)         │
                    └──────────────────────────┘
``` -->

**Data flow**: `Incoming data → Drift check → [if drifted] → Retrain → Eval gates → Promote → Hot-reload API → Serve`

---

## What It Does

| Capability | Description |
|---|---|
| **Model Training** | Trains a RandomForestRegressor on housing data, tracks experiments with MLflow, and versions every model (v1, v2, ...) |
| **Real-Time Predictions** | FastAPI server exposes a `/predict` endpoint; p95 < 15 ms measured |
| **Prediction Logging** | Every prediction stored in SQLite with input features, output, model version, and timestamp |
| **Observability** | `/metrics` endpoint exposes p50/p95/p99 latency, error rates, and SLO breach counts per endpoint |
| **Data Drift Detection** | KS-test + PSI per feature; overall drift flagged when ≥ 25% of features drift |
| **Automatic Retraining** | Drift triggers retrain → shadow evaluation → 3-gate quality check → promotion |
| **Eval Gates** | Gate 1: R2 ≥ 0.70 (absolute floor). Gate 2: R2 > production + margin (relative). Gate 3: RMSE ≤ 1.0 (ceiling) |
| **Model Rollback** | `/rollback` API endpoint to revert to any previous model version |
| **Streaming Simulation** | Gradual concept + covariate drift with resource-aware retraining (capped CPU, max retrains) |
| **Live Dashboard** | 4-tab Streamlit UI: Monitor, Predict, Drift Analysis, Operations |
| **CI/CD** | GitHub Actions: lint → train → test → drift check → API smoke test → Docker build |
| **Centralized Logging** | All modules log to console + `logs/mlops.log` with structured timestamps |

---

## How It Works

The system follows a closed-loop MLOps lifecycle:

1. **Train** — The training pipeline loads the California Housing dataset (20,640 samples, 8 features), preprocesses with StandardScaler, trains a RandomForest (100 trees, max_depth=10), evaluates (R2 = 0.894, RMSE = 0.509), logs to MLflow, and saves versioned artifacts.

2. **Serve** — FastAPI + Uvicorn serve predictions at p95 < 15 ms. Every request passes through `LatencyMiddleware` which tracks per-endpoint latency percentiles and SLO breaches. Predictions are logged to SQLite.

3. **Monitor** — Streamlit dashboard shows real-time metrics, prediction trends, feature distributions, and streaming simulation controls.

4. **Detect Drift** — Two statistical tests per feature:
   - **KS-test**: p-value < 0.05 → distribution change detected
   - **PSI**: > 0.2 → population shifted significantly
   - Overall drift flagged when ≥ 25% of features (2+ out of 8) drift

5. **Self-Heal** — When drift is detected, the retrain pipeline:
   - Combines reference + incoming labeled data
   - Trains a new model
   - Runs 3 eval gates (shadow test): R2 floor, relative improvement, RMSE ceiling
   - Promotes only if all gates pass
   - Hot-reloads the running API (zero-downtime model swap)

6. **Rollback** — If a deployed model underperforms, `POST /rollback?target_version=N` instantly reverts to a known-good version.

---

<!-- ## Architecture -->


## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Framework | scikit-learn | RandomForestRegressor training + inference |
| Experiment Tracking | MLflow | Log params, metrics, artifacts per run |
| API Server | FastAPI + Uvicorn | REST endpoints for prediction + model management |
| Observability | Custom middleware | p50/p95/p99 latency, SLO tracking, `/metrics` endpoint |
| Dashboard | Streamlit | Interactive monitoring, prediction, drift visualization |
| Drift Detection | scipy (KS-test) + custom PSI | Statistical distribution comparison |
| Database | SQLite | Prediction logging and history |
| Visualization | matplotlib | Overlaid reference vs. current distribution plots |
| Containerization | Docker + Docker Compose | Multi-service deployment |
| CI/CD | GitHub Actions | Test → smoke test → Docker build on push |
| Testing | pytest | 16 tests across API, drift, and training modules |
| Logging | Python logging | Centralized console + file logging |

---

## Project Structure

```
drift_detection_MLOPs/
├── api/                        # FastAPI inference server
│   ├── main.py                 #   App entry point, routes, CORS, lifespan
│   ├── models.py               #   Pydantic request/response schemas
│   ├── predictor.py            #   Model loading, scaling, inference
│   ├── database.py             #   SQLite connection, prediction logging
│   └── middleware.py           #   LatencyMiddleware, MetricsCollector, SLOs
│
├── training/                   # ML training pipeline
│   ├── config.py               #   Hyperparameters, paths, constants
│   ├── preprocess.py           #   Data loading, StandardScaler, train/test split
│   ├── train.py                #   Training loop + MLflow experiment tracking
│   ├── evaluate.py             #   R2, RMSE, MAE calculation
│   └── retrain_pipeline.py     #   Auto-retrain orchestrator (drift→eval gates→promote)
│
├── drift/                      # Drift detection module
│   ├── drift_check.py          #   KS-test + PSI per feature, combined check
│   ├── simulate_drift.py       #   Generate no/mild/heavy drifted datasets
│   └── stream_synthetic.py     #   Gradual concept+covariate drift simulator
│
├── dashboard/                  # Streamlit monitoring dashboard
│   └── app.py                  #   4-tab UI: Monitor, Predict, Drift Analysis, Operations
│
├── registry/                   # Model version management
│   └── promote_model.py        #   Find best model, promote to production/
│
├── utils/                      # Shared utilities
│   └── logging_config.py       #   Centralized logging (console + file)
│
├── tests/                      # Test suite (16 tests)
│   ├── test_api.py             #   API endpoint tests (health, predict)
│   ├── test_drift.py           #   Drift detection tests (PSI, KS, combined)
│   └── test_training.py        #   Training pipeline tests (data, model, eval)
│
├── models/                     # Trained model artifacts
│   ├── v1/, v2/, v3/ ...       #   Versioned: model.joblib + scaler + metadata.json
│   └── production/             #   Current production model
│
├── data/                       # Datasets and prediction database
│   ├── reference_data.csv      #   Training distribution for drift comparison
│   ├── drifted_data.csv        #   Simulated drifted data (for testing)
│   └── predictions.db          #   SQLite prediction log
│
├── logs/                       # Application logs
│   └── mlops.log               #   Structured log: timestamp | level | module | msg
│
├── docker/                     # Dockerfiles
│   ├── Dockerfile.api          #   API server container
│   └── Dockerfile.dashboard    #   Dashboard container
│
├── .github/workflows/          # CI/CD
│   └── ci.yml                  #   GitHub Actions: test→smoke→build
│
├── assets/                     # Screenshots for documentation
├── docker-compose.yml          #   Multi-service deployment (API + Dashboard)
├── requirements.txt            #   Python dependencies
├── pyproject.toml              #   Pytest configuration
├── conftest.py                 #   Pytest fixtures + warning suppression
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.11)
- pip
- (Optional) Docker and Docker Compose for containerized deployment

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/faisal-titu/drift_detection_MLOPs.git
cd drift_detection_MLOPs

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Initial Setup (Train + Promote)

Before using the API or dashboard, you need a trained production model:

```bash
# Train the model (creates versioned model + logs to MLflow)
python -m training.train

# Promote the best model to production
python -m registry.promote_model
```

You should see output like:
```
2026-02-15 11:00:00 | INFO     | training.train           | Training RandomForest model... (n_jobs=-1)
2026-02-15 11:00:01 | INFO     | training.train           | Model trained successfully
2026-02-15 11:00:01 | INFO     | training.train           | Model saved: models/v1/model.joblib
2026-02-15 11:00:02 | INFO     | registry.promote_model   | Model v1 promoted to production
```

---

## Usage Guide

### 1. Training a Model

```bash
python -m training.train
```

This will:
- Load the California Housing dataset (20,640 samples, 8 features)
- Split into 80% train / 20% test
- Scale features with StandardScaler
- Train a RandomForestRegressor (100 trees, max_depth=10)
- Evaluate on test set (R2 ≈ 0.894, RMSE ≈ 0.509, MAE ≈ 0.317)
- Log all parameters and metrics to MLflow
- Save versioned model to `models/v<N>/`
- Save reference data to `data/reference_data.csv` for drift detection

To view MLflow experiment tracking:
```bash
mlflow ui
# Open http://localhost:5000
```

---

### 2. Starting the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API server will:
- Load the production model from `models/production/`
- Start accepting prediction requests on port 8000
- Track per-request latency and SLO compliance via `LatencyMiddleware`
- Auto-initialize the SQLite database at `data/predictions.db`

API documentation is available at: **http://localhost:8000/docs**

---

### 3. Making Predictions

**Via curl:**
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

**Response:**
```json
{
  "prediction": 4.31,
  "model_version": 1,
  "timestamp": "2026-02-15T11:05:00",
  "prediction_id": 1
}
```

> The prediction value is in units of $100,000. So `4.31` = **$431,000**.

**Response header** includes `X-Response-Time-Ms` for client-side latency visibility.

**Via Python:**
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "MedInc": 5.0, "HouseAge": 30, "AveRooms": 6.0,
    "AveBedrms": 1.0, "Population": 1500, "AveOccup": 3.0,
    "Latitude": 34.0, "Longitude": -118.0,
})
print(response.json())
print(f"Latency: {response.headers['X-Response-Time-Ms']} ms")
```

**Feature descriptions:**

| Feature | Description | Typical Range |
|---|---|---|
| `MedInc` | Median income in block group (x$10K) | 0.5 - 15.0 |
| `HouseAge` | Median house age in years | 1 - 52 |
| `AveRooms` | Average rooms per household | 1 - 15 |
| `AveBedrms` | Average bedrooms per household | 0.5 - 5 |
| `Population` | Block group population | 1 - 40,000 |
| `AveOccup` | Average household occupancy | 1 - 10 |
| `Latitude` | Latitude coordinate | 32 - 42 |
| `Longitude` | Longitude coordinate | -125 to -114 |

---

### 4. Monitoring Dashboard

```bash
# In a separate terminal (API must be running)
streamlit run dashboard/app.py
# Open http://localhost:8501
```

The dashboard has four tabs:

| Tab | What It Shows |
|---|---|
| **Monitor** | Total predictions count, avg/min/max values, prediction distribution histogram, time series chart, feature statistics, and a scrollable predictions table |
| **Predict** | Interactive slider-based form for all 8 features. Submit to call the API and see the predicted house value, model version, and prediction ID |
| **Drift Analysis** | Click "Run Drift Analysis" to compare recent predictions against training data. Shows: drift status banner, summary metrics, overlaid reference vs. current distribution plots (blue/red histograms in a 2x4 grid), KS statistic and PSI score bar charts, and a detailed per-feature results table |
| **Operations** | Streaming simulation controls with live progress, drift intensity tracking, event log, start/stop buttons, and configurable parameters (batches, cooldown, max retrains, ramp rate) |

---

### 5. Drift Detection

**Check drift with your own data:**
```bash
python -m drift.drift_check --file data/your_data.csv
```

**Generate test datasets with controlled drift levels:**
```bash
python -m drift.simulate_drift
```

This creates three files in `data/`:
| File | Description |
|---|---|
| `no_drift_data.csv` | Sampled from training distribution (no shift) |
| `mild_drift_data.csv` | 2 features shifted slightly |
| `drifted_data.csv` | 6 features shifted heavily |

**Drift detection methods:**

| Method | What It Measures | Threshold | Interpretation |
|---|---|---|---|
| **KS-test** | Max difference between two CDFs | p-value < 0.05 | Distributions are statistically different |
| **PSI** | Shift in binned distributions | PSI > 0.2 | Population has changed significantly |

Overall drift is triggered when **≥ 25%** of features (2+ out of 8) show drift by either method.

---

### 6. Auto-Retraining

The self-healing pipeline detects drift and automatically retrains:

![Auto-Retraining](assets/retrain.png)

**Run with simulated drift:**
```bash
python -m training.retrain_pipeline --simulate
```

**Run with real incoming data:**
```bash
python -m training.retrain_pipeline --data path/to/incoming_data.csv
```

**Force retrain (skip drift check):**
```bash
python -m training.retrain_pipeline --force
```

**Promote only if the new model improves over production:**
```bash
python -m training.retrain_pipeline --data path/to/data.csv --min-r2-improvement 0.02
```

The pipeline runs 3 eval gates before promotion:
1. **R2 ≥ 0.70** — absolute quality floor
2. **new_R2 ≥ current_R2 + margin** — must actually improve
3. **RMSE ≤ 1.0** — error magnitude ceiling

Failed gates log `EVAL GATE FAIL` and keep the current production model.

---

### 7. Streaming Synthetic Concept Drift

For a realistic drift stress test, stream **labeled synthetic batches** where:
- Feature distributions shift gradually (**covariate drift** ramps 0% → 100%)
- Feature-target relationships change (**concept drift** blends stable → drifted)

```bash
python -m drift.stream_synthetic \
  --batches 20 \
  --batch-size 300 \
  --drift-start 8 \
  --ramp-batches 8 \
  --perf-r2-threshold 0.40 \
  --min-r2-improvement 0.02 \
  --cooldown 5 \
  --max-retrains 3 \
  --sleep 1.0
```

**Resource safeguards:**
- Drift ramps linearly over `--ramp-batches` (never spikes from 0→100% in one batch)
- Minimum 0.5s sleep enforced between batches
- Retraining caps `n_jobs=2` (won't saturate all CPU cores)
- Hard `--max-retrains` cap prevents infinite retrain loops

---

### 8. Observability & Metrics

After the API is running, hit the metrics endpoint:

```bash
curl http://localhost:8000/metrics | python -m json.tool
```

**Response example:**
```json
{
  "/predict": {
    "total_requests": 150,
    "error_count": 0,
    "error_rate": 0.0,
    "slo_target_ms": 150,
    "slo_breaches": 0,
    "slo_breach_rate": 0.0,
    "p50_ms": 8.42,
    "p95_ms": 14.31,
    "p99_ms": 22.87,
    "avg_ms": 9.15,
    "max_ms": 35.10
  },
  "/health": {
    "total_requests": 50,
    "p50_ms": 0.45,
    "p95_ms": 1.20,
    "slo_target_ms": 50,
    "slo_breaches": 0
  },
  "_global": {
    "total_requests": 200,
    "error_count": 0,
    "p50_ms": 5.21,
    "p95_ms": 13.80,
    "p99_ms": 22.10
  }
}
```

SLO breaches are also logged as warnings in `logs/mlops.log`:
```
2026-02-25 14:30:01 | WARNING  | api.middleware | SLO breach: POST /predict took 163.2 ms (target: 150 ms)
```

---

### 9. Model Rollback

If a newly promoted model underperforms in production, instantly roll back:

```bash
# Roll back to model v2
curl -X POST "http://localhost:8000/rollback?target_version=2"
```

**Response:**
```json
{
  "status": "rolled_back",
  "model_version": 2,
  "r2": 0.8812
}
```

This promotes the specified version to `models/production/` and hot-reloads the serving model with zero downtime.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root - returns API info |
| `GET` | `/health` | Health check (model status, version) |
| `POST` | `/predict` | Make a prediction (JSON body → price) |
| `GET` | `/predictions?limit=100` | Get recent prediction history |
| `POST` | `/reload` | Hot-reload the production model |
| `GET` | `/metrics` | Latency percentiles, error rates, SLO status |
| `POST` | `/rollback?target_version=N` | Roll back to a previous model version |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## Docker Deployment

Build and run both services with Docker Compose:

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

| Service | Container | Port | URL |
|---|---|---|---|
| API Server | `mlops-api` | 8000 | http://localhost:8000 |
| Dashboard | `mlops-dashboard` | 8501 | http://localhost:8501 |

Health check configured: Docker restarts the API container if `/health` fails 3 times in a row (30s interval).

To stop:
```bash
docker compose down
```

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main`/`RnD` and PRs to `main`:

```
┌──────────────────────────────────────────────────────┐
│  test job                                            │
│  ├─ Checkout → Setup Python 3.11 → Install deps      │
│  ├─ Train initial model                              │
│  ├─ Promote model to production                      │
│  ├─ Run 16 pytest tests                              │
│  ├─ Run drift check (no-drift scenario)              │
│  └─ API smoke test:                                  │
│     ├─ Start Uvicorn                                 │
│     ├─ Verify /health (model loaded)                 │
│     ├─ Verify /predict (valid response)              │
│     └─ Verify /metrics (latency tracking active)     │
│                                                      │
│  build job (main branch only)                        │
│  ├─ Build API Docker image                           │
│  └─ Build Dashboard Docker image                     │
└──────────────────────────────────────────────────────┘
```

---

## Running Tests

```bash
# Run all 16 tests
pytest tests/ -v

# Run specific test module
pytest tests/test_api.py -v
pytest tests/test_drift.py -v
pytest tests/test_training.py -v
```

**Test coverage:**

| Module | Tests | What It Validates |
|---|---|---|
| `test_api.py` | 4 | Health endpoint returns 200, predict with valid/invalid input |
| `test_drift.py` | 6 | PSI calculation (same/different distributions), KS-test (no/heavy drift), combined check |
| `test_training.py` | 6 | Data loading, preprocessing shapes, scaler output, model training, evaluation metrics, acceptance threshold |

---

## Postmortem & Lessons Learned

### Incident 1: Streaming Simulator Saturated CPU & RAM

**What happened**: The synthetic drift simulator (`stream_synthetic.py`) with default settings (batch_size=1000, sleep=0s, cooldown=2) caused 100% CPU usage and memory exhaustion. Every batch after drift_start triggered retrain because all 8 features drifted simultaneously (features were multiplied by 1.5–1.8x in one step). With `n_jobs=-1`, each retrain used all cores.

**Root cause**: Catastrophic (non-gradual) drift + no resource limits + no sleep between batches + low retrain cooldown.

**Fix applied**:
1. Changed drift from binary to **gradual ramp** (0% → 100% over configurable `ramp_batches`)
2. Reduced covariate shift multipliers from 1.8x to ~1.12x
3. Enforced minimum 0.5s sleep between batches
4. Added `max_retrains` hard cap (default: 3)
5. Added `DRIFT_RETRAIN_N_JOBS` env var to cap retrain CPU usage to 2 cores
6. Raised default cooldown from 2 → 5 batches

**Result**: Stable simulation — batches 0–7 show no drift, 8–11 ramp gradually, 12+ show detectable but not catastrophic drift. CPU stays under 50%.

### Incident 2: Retrain Loop Deploys Worse Models

**What happened**: With `min_r2_improvement=0.0`, the retrain pipeline would sometimes promote models that scored marginally better on the combined train+drift data but performed worse on clean holdout data.

**Root cause**: No RMSE ceiling check; relative improvement threshold set to zero.

**Fix applied**:
1. Added **3-gate eval system**: absolute R2 floor (0.70), relative improvement margin, and RMSE ceiling (1.0)
2. Default `min_r2_improvement` raised to 0.02
3. All gate failures log `EVAL GATE FAIL` for easy alerting

**Lesson**: Always run a shadow test / eval gate before promoting a retrained model. "Better R2 on mixed data" doesn't guarantee better production performance.

---

## Screenshots

### Streamlit Dashboard - Monitor Tab
![Dashboard](assets/dashboard.png)

### Streamlit Dashboard - Predict Tab
![Predict Tab](assets/predict_tab.png)

### Streamlit Dashboard - Drift Analysis
![Drift Analysis](assets/drift_analysis.png)

### Drift Distribution Charts
![Drift Charts](assets/drift_charts.png)

### FastAPI Documentation (Swagger UI)
![API Docs](assets/api_docs.png)

---

## Logging

All modules use centralized logging configured in `utils/logging_config.py`:

- **Console**: INFO-level messages printed to stdout
- **File**: DEBUG-level messages written to `logs/mlops.log`

Log format:
```
2026-02-15 11:00:01 | INFO     | training.train           | Model trained successfully
2026-02-15 11:00:02 | WARNING  | drift.drift_check        | Drift detected in 5/8 features
2026-02-15 11:00:03 | WARNING  | api.middleware            | SLO breach: POST /predict took 163.2 ms (target: 150 ms)
2026-02-15 11:00:04 | ERROR    | training.retrain_pipeline | EVAL GATE FAIL: RMSE 1.23 exceeds 1.0 ceiling
```

---

## License

MIT

---
