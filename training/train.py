"""
Model Training Module with MLflow Tracking
Trains and saves the ML model with versioning and experiment tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import (
    MODEL_PARAMS,
    MODELS_DIR,
    FEATURE_NAMES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    get_next_version,
    get_model_version_path,
    get_effective_model_params,
)
from .preprocess import (
    load_data,
    preprocess_data,
    save_reference_data,
    save_scaler,
)
from .evaluate import evaluate_model, print_metrics, is_model_acceptable
from utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Train a RandomForest model.

    Uses ``get_effective_model_params()`` so that the ``DRIFT_RETRAIN_N_JOBS``
    env var is respected at runtime (set by the streaming simulator to cap
    CPU usage during automated retrains).
    """
    params = get_effective_model_params()
    logger.info("Training RandomForest model... (n_jobs=%s)", params.get("n_jobs"))

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    logger.info("Model trained successfully")
    return model


def save_model(
    model: RandomForestRegressor,
    version: int,
    metrics: Dict[str, float],
    feature_names: list = None
) -> Path:
    """
    Save model with metadata.

    Args:
        model: Trained model
        version: Model version number
        metrics: Performance metrics
        feature_names: List of feature names

    Returns:
        Path to saved model directory
    """
    model_dir = get_model_version_path(version)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_names": feature_names or FEATURE_NAMES,
        "model_params": MODEL_PARAMS,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved: %s", model_path)
    logger.info("Metadata saved: %s", metadata_path)

    return model_dir


def load_model(model_dir: Path) -> RandomForestRegressor:
    """
    Load a saved model.

    Args:
        model_dir: Directory containing the model

    Returns:
        Loaded model
    """
    model_path = model_dir / "model.joblib"
    return joblib.load(model_path)


def train_with_mlflow() -> Tuple[RandomForestRegressor, Dict[str, float], int]:
    """
    Full training pipeline with MLflow tracking.

    Returns:
        Tuple of (model, metrics, version)
    """
    setup_mlflow()

    with mlflow.start_run() as run:
        logger.info("MLflow Run ID: %s", run.info.run_id[:8])

        mlflow.log_params(MODEL_PARAMS)

        # Step 1: Load data
        X, y = load_data()

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

        # Step 3: Save reference data for drift detection
        save_reference_data(X, y)

        # Step 4: Train model
        model = train_model(X_train, y_train)

        # Step 5: Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)

        print_metrics(train_metrics, "Training Metrics")
        print_metrics(test_metrics, "Test Metrics")

        # Step 6: Check if model is acceptable
        if is_model_acceptable(test_metrics):
            logger.info("Model meets performance criteria")
        else:
            logger.warning("Model below performance threshold (R2 < 0.7)")

        # Step 7: Save model with new version
        version = get_next_version()
        model_dir = save_model(model, version, test_metrics)
        save_scaler(scaler, model_dir)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_version", version)

        logger.info("MLflow Run: %s", run.info.run_id)

    return model, test_metrics, version


def retrain_with_data(
    X: "pd.DataFrame",
    y: "pd.Series",
    run_name: str = "retrain",
) -> Tuple[RandomForestRegressor, Dict[str, float], int]:
    """
    Retrain model with custom data (used by auto-retraining pipeline).

    Args:
        X: Feature DataFrame
        y: Target Series
        run_name: MLflow run name

    Returns:
        Tuple of (model, metrics, version)
    """
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("MLflow Run ID: %s", run.info.run_id[:8])

        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("retrain", True)
        mlflow.log_param("data_size", len(X))

        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        save_reference_data(X, y)

        model = train_model(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)

        print_metrics(test_metrics, "Retrain Test Metrics")

        version = get_next_version()
        model_dir = save_model(model, version, test_metrics)
        save_scaler(scaler, model_dir)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_version", version)

    return model, test_metrics, version


def main():
    """Main training pipeline with MLflow."""
    logger.info("=" * 50)
    logger.info("MLOps Training Pipeline + MLflow")
    logger.info("=" * 50)

    model, test_metrics, version = train_with_mlflow()

    logger.info("=" * 50)
    logger.info("Training Complete - Model v%d", version)
    logger.info("  R2 Score: %.4f", test_metrics['r2'])
    logger.info("=" * 50)

    return model, test_metrics, version


if __name__ == "__main__":
    main()
