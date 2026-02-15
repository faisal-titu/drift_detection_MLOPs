"""
Model Evaluation Module
Calculates and displays model performance metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

from utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Model Metrics") -> None:
    """
    Log metrics in a formatted table.

    Args:
        metrics: Dictionary of metric names to values
        title: Title for the metrics display
    """
    metric_names = {
        "mse": "Mean Squared Error",
        "rmse": "Root Mean Squared Error",
        "mae": "Mean Absolute Error",
        "r2": "R2 Score",
    }

    logger.info("=" * 40)
    logger.info("%s", title)
    logger.info("=" * 40)

    for key, value in metrics.items():
        name = metric_names.get(key, key)
        logger.info("  %s: %.4f", name, value)

    logger.info("=" * 40)


def is_model_acceptable(metrics: Dict[str, float], r2_threshold: float = 0.7) -> bool:
    """
    Check if model meets minimum performance criteria.

    Args:
        metrics: Dictionary of metrics
        r2_threshold: Minimum acceptable R2 score

    Returns:
        True if model is acceptable
    """
    return metrics.get("r2", 0) >= r2_threshold


if __name__ == "__main__":
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])

    metrics = evaluate_model(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
    logger.info("Model acceptable: %s", is_model_acceptable(metrics))
