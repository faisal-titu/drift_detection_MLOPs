"""
Data Preprocessing Module
Handles data loading, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple

from .config import (
    FEATURE_NAMES,
    TARGET_NAME,
    TEST_SIZE,
    RANDOM_STATE,
    DATA_DIR,
    REFERENCE_DATA_PATH,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    housing = fetch_california_housing()

    X = pd.DataFrame(housing.data, columns=FEATURE_NAMES)
    y = pd.Series(housing.target, name=TARGET_NAME)

    logger.info("Loaded dataset: %d samples, %d features", X.shape[0], X.shape[1])
    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess data: scale features and split into train/test.

    Args:
        X: Feature DataFrame
        y: Target Series
        scaler: Optional pre-fitted scaler
        fit_scaler: Whether to fit the scaler (True for training)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    logger.info("Preprocessed: %d train, %d test samples", len(X_train), len(X_test))

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


def save_reference_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Save training data distribution for drift detection.

    Args:
        X: Feature DataFrame
        y: Target Series
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = X.copy()
    reference_df[TARGET_NAME] = y.values

    reference_df.to_csv(REFERENCE_DATA_PATH, index=False)
    logger.info("Reference data saved: %s", REFERENCE_DATA_PATH)


def save_scaler(scaler: StandardScaler, model_dir: Path) -> None:
    """
    Save the fitted scaler alongside the model.

    Args:
        scaler: Fitted StandardScaler
        model_dir: Directory to save the scaler
    """
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved: %s", scaler_path)


def load_scaler(model_dir: Path) -> StandardScaler:
    """
    Load a saved scaler.

    Args:
        model_dir: Directory containing the scaler

    Returns:
        Fitted StandardScaler
    """
    scaler_path = model_dir / "scaler.joblib"
    return joblib.load(scaler_path)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    save_reference_data(X, y)

    logger.info("Data statistics:")
    logger.info("  X_train shape: %s", X_train.shape)
    logger.info("  X_test shape:  %s", X_test.shape)
    logger.info("  y_train range: [%.2f, %.2f]", y_train.min(), y_train.max())
