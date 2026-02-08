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
)
from .preprocess import (
    load_data,
    preprocess_data,
    save_reference_data,
    save_scaler,
)
from .evaluate import evaluate_model, print_metrics, is_model_acceptable


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"📊 MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Train a RandomForest model.
    
    Args:
        X_train: Training features
        y_train: Training targets
    
    Returns:
        Trained model
    """
    print("🚀 Training RandomForest model...")
    
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully")
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
    
    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    
    # Save metadata
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
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
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
        print(f"� MLflow Run ID: {run.info.run_id[:8]}...")
        
        # Log parameters
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
        
        # Log metrics to MLflow
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        
        print_metrics(train_metrics, "Training Metrics")
        print_metrics(test_metrics, "Test Metrics")
        
        # Step 6: Check if model is acceptable
        if is_model_acceptable(test_metrics):
            print("✅ Model meets performance criteria!")
        else:
            print("⚠️ Model below performance threshold (R² < 0.7)")
        
        # Step 7: Save model with new version
        version = get_next_version()
        model_dir = save_model(model, version, test_metrics)
        save_scaler(scaler, model_dir)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_version", version)
        
        print(f"\n📊 MLflow Run: {run.info.run_id}")
        
    return model, test_metrics, version


def main():
    """Main training pipeline with MLflow."""
    print("\n" + "="*50)
    print("🎯 MLOps Training Pipeline + MLflow")
    print("="*50 + "\n")
    
    model, test_metrics, version = train_with_mlflow()
    
    print("\n" + "="*50)
    print(f"🎉 Training Complete! Model v{version}")
    print(f"   R² Score: {test_metrics['r2']:.4f}")
    print("="*50 + "\n")
    
    return model, test_metrics, version


if __name__ == "__main__":
    main()
