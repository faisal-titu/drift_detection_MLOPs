# Training Module
"""ML training pipeline: preprocessing, training, and evaluation."""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    FEATURE_NAMES,
    TARGET_NAME,
    MODEL_PARAMS,
    get_model_version_path,
    get_latest_version,
    get_next_version,
)
from .preprocess import (
    load_data,
    preprocess_data,
    save_reference_data,
    save_scaler,
    load_scaler,
)
from .train import train_model, save_model, load_model, main
from .evaluate import evaluate_model, print_metrics, is_model_acceptable

__all__ = [
    # Config
    "PROJECT_ROOT",
    "DATA_DIR", 
    "MODELS_DIR",
    "FEATURE_NAMES",
    "TARGET_NAME",
    "MODEL_PARAMS",
    "get_model_version_path",
    "get_latest_version",
    "get_next_version",
    # Preprocess
    "load_data",
    "preprocess_data",
    "save_reference_data",
    "save_scaler",
    "load_scaler",
    # Train
    "train_model",
    "save_model",
    "load_model",
    "main",
    # Evaluate
    "evaluate_model",
    "print_metrics",
    "is_model_acceptable",
]
