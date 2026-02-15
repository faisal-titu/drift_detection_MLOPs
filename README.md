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

[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github)](/.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
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
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Running Tests](#running-tests)
- [Screenshots](#screenshots)
- [License](#license)

---

## What It Does

| Capability | Description |
|---|---|
| **Model Training** | Trains a RandomForestRegressor on housing data, tracks experiments with MLflow, and versions every model (v1, v2, ...) |
| **Real-Time Predictions** | FastAPI server exposes a `/predict` endpoint that accepts JSON input and returns house price predictions instantly |
| **Prediction Logging** | Every prediction is stored in a SQLite database with input features, output, model version, and timestamp |
| **Data Drift Detection** | Compares incoming prediction data against the training distribution using two statistical tests: Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI) |
| **Automatic Retraining** | When drift is detected across >= 25% of features, the pipeline retrains the model, evaluates it, and promotes it to production — all without manual intervention |
| **Live Dashboard** | Streamlit app with three tabs: Monitor (metrics/charts), Predict (slider-based input form), and Drift Analysis (overlaid distribution plots) |
| **Centralized Logging** | All modules log to both console and `logs/mlops.log` with structured timestamps, levels, and module names |

---

## How It Works

The system follows a closed-loop MLOps lifecycle:

1. **Train** — The training pipeline loads the California Housing dataset, preprocesses features (StandardScaler), trains a RandomForest model, evaluates it (R2, RMSE, MAE), logs everything to MLflow, and saves a versioned model artifact.

2. **Serve** — The FastAPI server loads the production model and scaler, accepts prediction requests via REST API, scales the input features, runs inference, and logs every prediction to SQLite.

3. **Monitor** — The Streamlit dashboard fetches predictions from the database and displays real-time metrics, distribution charts, and feature summaries.

4. **Detect Drift** — The drift module loads the training reference data and compares it against recent predictions using two tests:
   - **KS-test**: Two-sample Kolmogorov-Smirnov test (null hypothesis: same distribution). Drift if p-value < 0.05.
   - **PSI**: Population Stability Index (measures shift in binned distributions). Drift if PSI > 0.2.
   - Overall drift is flagged when >= 25% of features drift by either method.

5. **Self-Heal** — When drift is detected, the auto-retraining pipeline: combines reference + incoming data, retrains a new model, checks if R2 >= 0.7, promotes it to production, and hot-reloads the running API server.

---

## Architecture

```
+---------------------------------------------------------------------+
|                    MLOps Pipeline Architecture                      |
+---------------------------------------------------------------------+
|
|   TRAINING PIPELINE                                                 
|   ================                                                  
|   California Housing Dataset                                        
|         |                                                           
|         v                                                           
|   Preprocess + StandardScaler                                       
|         |                                                           
|         v                                                           
|   Train RandomForest ---------> Log to MLflow (params, metrics)     
|         |                                                           
|         v                                                           
|   Evaluate (R2, RMSE, MAE)                                          
|         |                                                           
|         v                                                           
|   Save Versioned Model (models/v1, v2, ...)                         
|         |                                                           
|         v                                                           
|   INFERENCE SERVER                MONITORING DASHBOARD              
|   ================                ====================              
|   Production Model                Streamlit :8501                   
|         |                              |                            
|         v                         +----+----+--------+              
|   FastAPI :8000                   |         |        |              
|     |       |                   Monitor  Predict  Drift Analysis    
|     |       v                   (charts)  (sliders) (distributions) 
|     |    SQLite DB  ----------------^                               
|     |       |                                                       
|     |       v                                                       
|     |    DRIFT DETECTION + SELF-HEALING                             
|     |    ==================================                         
|     |    Drift Check (KS-test + PSI)                                
|     |       |                |                                      
|     |       v                v                                      
|     |    No Drift          Drift Detected                           
|     |    (skip)               |                                     
|     |                        v                                      
|     |                   Retrain Model                               
|     |                        |                                      
|     |                        v                                      
|     |                   Evaluate (R2 >= 0.7?)                       
|     |                        |                                      
|     |                        v                                      
|     |                   Promote to Production                       
|     |                        |                                      
|     |                        v                                      
|     | <---- Hot-reload via POST /reload                              
|                                                                     
+---------------------------------------------------------------------+
|   Docker Compose      GitHub Actions CI      Hugging Face Spaces    |
+---------------------------------------------------------------------+
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Framework | scikit-learn | RandomForestRegressor training + inference |
| Experiment Tracking | MLflow | Log params, metrics, artifacts per run |
| API Server | FastAPI + Uvicorn | REST endpoints for prediction + model management |
| Dashboard | Streamlit | Interactive monitoring, prediction, drift visualization |
| Drift Detection | scipy (KS-test) + custom PSI | Statistical distribution comparison |
| Database | SQLite | Prediction logging and history |
| Visualization | matplotlib | Overlaid reference vs. current distribution plots |
| Containerization | Docker + Docker Compose | Multi-service deployment |
| CI/CD | GitHub Actions | Automated testing and Docker build on push |
| Testing | pytest | 16 tests across API, drift, and training modules |
| Logging | Python logging | Centralized console + file logging |

---

## Project Structure

```
drift_detection_MLOPs/
├── api/                        # FastAPI inference server
│   ├── __init__.py
│   ├── main.py                 #   App entry point, routes, CORS, lifespan
│   ├── models.py               #   Pydantic request/response schemas
│   ├── predictor.py            #   Model loading, scaling, inference
│   └── database.py             #   SQLite connection, prediction logging
│
├── training/                   # ML training pipeline
│   ├── __init__.py
│   ├── config.py               #   Hyperparameters, paths, constants
│   ├── preprocess.py           #   Data loading, StandardScaler, train/test split
│   ├── train.py                #   Training loop + MLflow experiment tracking
│   ├── evaluate.py             #   R2, RMSE, MAE calculation
│   └── retrain_pipeline.py     #   Auto-retrain orchestrator (drift->train->promote)
│
├── drift/                      # Drift detection module
│   ├── __init__.py
│   ├── drift_check.py          #   KS-test + PSI per feature, combined check
│   └── simulate_drift.py       #   Generate no/mild/heavy drifted datasets
│
├── dashboard/                  # Streamlit monitoring dashboard
│   └── app.py                  #   3-tab UI: Monitor, Predict, Drift Analysis
│
├── registry/                   # Model version management
│   └── promote_model.py        #   Find best model, promote to production/
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   └── logging_config.py       #   Centralized logging (console + file)
│
├── tests/                      # Test suite (16 tests)
│   ├── test_api.py             #   API endpoint tests (health, predict)
│   ├── test_drift.py           #   Drift detection tests (PSI, KS, combined)
│   └── test_training.py        #   Training pipeline tests (data, model, eval)
│
├── models/                     # Trained model artifacts (git-ignored)
│   ├── v1/, v2/, v3/           #   Versioned: model.joblib + metadata.json
│   └── production/             #   Symlinked production model
│
├── data/                       # Datasets and prediction database
│   ├── reference_data.csv      #   Training distribution for drift comparison
│   ├── drifted_data.csv        #   Simulated drifted data (for testing)
│   └── predictions.db          #   SQLite prediction log
│
├── logs/                       # Application logs (git-ignored)
│   └── mlops.log               #   Structured log: timestamp | level | module | msg
│
├── docker/                     # Dockerfiles
│   ├── Dockerfile.api          #   API server container
│   └── Dockerfile.dashboard    #   Dashboard container
│
├── .github/workflows/          # CI/CD
│   └── ci.yml                  #   GitHub Actions: lint, test, docker build
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
git clone https://github.com/<your-username>/drift_detection_MLOPs.git
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
2026-02-15 11:00:00 | INFO     | training.train           | Training RandomForest model...
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
- Train a RandomForestRegressor (100 trees, max_depth=15)
- Evaluate on test set (R2, RMSE, MAE)
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

**Via Python:**
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "MedInc": 5.0, "HouseAge": 30, "AveRooms": 6.0,
    "AveBedrms": 1.0, "Population": 1500, "AveOccup": 3.0,
    "Latitude": 34.0, "Longitude": -118.0,
})
print(response.json())
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

The dashboard has three tabs:

| Tab | What It Shows |
|---|---|
| **Monitor** | Total predictions count, avg/min/max values, prediction distribution histogram, time series chart, feature statistics, and a scrollable predictions table |
| **Predict** | Interactive slider-based form for all 8 features. Submit to call the API and see the predicted house value, model version, and prediction ID |
| **Drift Analysis** | Click "Run Drift Analysis" to compare recent predictions against training data. Shows: drift status banner, summary metrics, overlaid reference vs. current distribution plots (blue/red histograms in a 2x4 grid), KS statistic and PSI score bar charts, and a detailed per-feature results table |

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

**Test drift detection on simulated data:**
```bash
# Test with no-drift data (should report no drift)
python -m drift.drift_check --no-drift

# Test with heavy-drift data
python -m drift.drift_check --file data/drifted_data.csv
```

**Drift detection methods:**

| Method | What It Measures | Threshold | Interpretation |
|---|---|---|---|
| **KS-test** | Max difference between two CDFs | p-value < 0.05 | Distributions are statistically different |
| **PSI** | Shift in binned distributions | PSI > 0.2 | Population has changed significantly |

Overall drift is triggered when **>= 25%** of features (2+ out of 8) show drift by either method.

---

### 6. Auto-Retraining

The self-healing pipeline detects drift and automatically retrains:

```
Load incoming data
    |
    v
Check for drift (KS + PSI)
    |
    +--> No drift --> Exit (no action needed)
    |
    v
Retrain model on combined data
    |
    v
Evaluate new model (R2 >= 0.7?)
    |
    +--> Below threshold --> Keep current model
    |
    v
Promote new model to production/
    |
    v
Hot-reload API server (/reload)
```

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

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root - returns API info |
| `GET` | `/health` | Health check (returns model status, version) |
| `POST` | `/predict` | Make a prediction (accepts JSON body, returns price) |
| `GET` | `/predictions?limit=100` | Get recent prediction history |
| `POST` | `/reload` | Hot-reload the production model |
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

To stop:
```bash
docker compose down
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
2026-02-15 11:00:03 | ERROR    | api.predictor            | Model not found: models/production/model.joblib
```

---

## License

MIT

---

## 🤗 Hugging Face Spaces Deployment

This project is configured for one-click deployment to Hugging Face Spaces using Docker.

1. **Create a Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Choose **Public** or **Private**

2. **Deploy Code**
   - Clone the Space repository to your machine
     ```bash
     git clone https://huggingface.co/spaces/<your-username>/<space-name>
     cd <space-name>
     ```
   - Copy all files from this project into the Space folder
     ```bash
     cp -r /path/to/drift_detection_MLOPs/* .
     ```
   - Add, commit, and push
     ```bash
     git add .
     git commit -m "Deploy to HF Spaces"
     git push
     ```

3. **Access App**
   - The Space will build automatically (takes ~2 mins)
   - Your app will be live at: `https://huggingface.co/spaces/<your-username>/<space-name>`
   - The dashboard and API will both be running in the same Space!
