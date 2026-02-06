# 🚀 MLOps Drift Detection System

A production-grade MLOps system featuring real-time prediction, automated drift detection, continuous retraining, and model versioning.

## 🎯 Features

- **Real-time Inference API** - FastAPI server for predictions
- **Monitoring Dashboard** - Streamlit UI for system visibility
- **Drift Detection** - Automatic data drift monitoring (KS-test/PSI)
- **Auto Retraining** - Self-healing ML pipeline
- **Model Registry** - Version control for ML models
- **Dockerized** - One-command deployment
- **CI/CD** - GitHub Actions pipeline

## 📁 Project Structure

```
├── data/                 # Raw and processed datasets
├── training/             # ML training pipeline
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── api/                  # FastAPI inference server
├── dashboard/            # Streamlit monitoring UI
├── drift/                # Drift detection logic
├── models/               # Trained models
│   ├── v1/, v2/, ...
│   └── production/       # Current production model
├── registry/             # Model promotion utilities
├── docker/               # Dockerfiles
├── .github/workflows/    # CI/CD pipelines
└── tests/                # Unit tests
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Dashboard | Streamlit |
| Drift Detection | scipy (KS-test, PSI) |
| Database | SQLite |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## 📊 Dataset

**California Housing Dataset** (built into scikit-learn)
- 20,640 samples
- 8 numeric features
- Regression task (predicting median house value)

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd drift_detection_MLOPs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train initial model
python training/train.py

# Start API server
uvicorn api.main:app --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py
```

## 🔄 MLOps Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Training │───▶│  Model   │───▶│   API    │───▶│Dashboard │
│ Pipeline │    │ Registry │    │ Server   │    │Monitoring│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
      ▲                               │
      │         ┌──────────┐          │
      └─────────│  Drift   │◀─────────┘
                │ Detection│
                └──────────┘
```

## 📈 Development Progress

- [x] Day 1: Project skeleton
- [ ] Day 2: Training pipeline
- [ ] Day 3: MLflow + versioning
- [ ] Day 4: FastAPI server
- [ ] Day 5: Streamlit dashboard
- [ ] Day 6: Drift detection
- [ ] Day 7: Auto retraining
- [ ] Day 8: Docker
- [ ] Day 9: CI/CD
- [ ] Day 10: Deployment

## 📄 License

MIT
