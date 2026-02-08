# API Module
"""FastAPI inference server for predictions."""

from .main import app
from .predictor import predictor, Predictor
from .database import log_prediction, get_predictions, get_prediction_count
from .models import PredictionRequest, PredictionResponse, HealthResponse

__all__ = [
    "app",
    "predictor",
    "Predictor",
    "log_prediction",
    "get_predictions",
    "get_prediction_count",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
]
