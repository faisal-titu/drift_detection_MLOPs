"""Tests for training module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from training.preprocess import load_data, preprocess_data
from training.evaluate import evaluate_model, is_model_acceptable
from training.train import train_model


class TestDataLoading:
    """Test data loading and preprocessing."""

    def test_load_data(self):
        X, y = load_data()
        assert X.shape[0] > 0
        assert X.shape[1] == 8
        assert len(y) == X.shape[0]

    def test_preprocess_data(self):
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        assert len(X_train) > len(X_test)
        assert X_train.shape[1] == 8
        assert scaler is not None

    def test_preprocess_scaling(self):
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # Scaled data should have mean ~0 and std ~1
        means = np.abs(X_train.mean(axis=0))
        assert all(m < 1.0 for m in means), "Scaled means should be near 0"


class TestModelTraining:
    """Test model training."""

    def test_train_model(self):
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        model = train_model(X_train, y_train)
        
        assert isinstance(model, RandomForestRegressor)
        preds = model.predict(X_test[:5])
        assert len(preds) == 5
        assert all(p > 0 for p in preds)


class TestEvaluation:
    """Test model evaluation."""

    def test_evaluate_model(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.9

    def test_is_model_acceptable(self):
        good_metrics = {"r2": 0.8, "mse": 0.1}
        bad_metrics = {"r2": 0.3, "mse": 1.0}
        
        assert is_model_acceptable(good_metrics) is True
        assert is_model_acceptable(bad_metrics) is False
