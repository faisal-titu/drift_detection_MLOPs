"""
Training Configuration
Centralized hyperparameters and paths for the ML pipeline.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_MODEL_DIR = MODELS_DIR / "production"
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.csv"

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "california_housing_regression"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"

# Model settings
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature names (California Housing)
FEATURE_NAMES = [
    "MedInc",      # Median income
    "HouseAge",    # Housing median age
    "AveRooms",    # Average rooms per household
    "AveBedrms",   # Average bedrooms per household
    "Population",  # Block group population
    "AveOccup",    # Average household members
    "Latitude",    # Block group latitude
    "Longitude",   # Block group longitude
]

TARGET_NAME = "MedHouseVal"


def get_model_version_path(version: int) -> Path:
    """Get path for a specific model version."""
    return MODELS_DIR / f"v{version}"


def get_latest_version() -> int:
    """Get the latest model version number."""
    if not MODELS_DIR.exists():
        return 0
    
    versions = []
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            versions.append(int(d.name[1:]))
    
    return max(versions) if versions else 0


def get_next_version() -> int:
    """Get the next model version number."""
    return get_latest_version() + 1
