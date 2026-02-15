"""
FastAPI Inference Server
Main application with prediction endpoint.
"""

import logging
import warnings
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import PredictionRequest, PredictionResponse, HealthResponse
from .database import log_prediction, get_predictions, get_prediction_count
from .predictor import predictor

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting MLOps Inference Server...")
    if not predictor.load_model():
        logger.warning("Model not loaded. Run training first.")
    yield
    logger.info("Shutting down server...")


app = FastAPI(
    title="MLOps Inference API",
    description="Real-time prediction API for California Housing prices",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "MLOps Inference API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded() else "degraded",
        model_loaded=predictor.is_loaded(),
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for housing price.

    The prediction is in units of $100,000 (e.g., 3.5 = $350,000).
    """
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a model is trained and promoted."
        )

    features = request.model_dump()

    try:
        prediction, model_version = predictor.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    timestamp = datetime.now()
    prediction_id = log_prediction(features, prediction, model_version)

    return PredictionResponse(
        prediction=prediction,
        model_version=model_version,
        timestamp=timestamp,
        prediction_id=prediction_id,
    )


@app.get("/predictions", tags=["Predictions"])
async def get_recent_predictions(limit: int = 100):
    """Get recent predictions."""
    predictions = get_predictions(limit)
    count = get_prediction_count()
    return {"total": count, "predictions": predictions}


@app.post("/reload", tags=["Admin"])
async def reload_model():
    """Reload the production model."""
    success = predictor.reload_model()
    if success:
        return {"status": "reloaded", "model_version": predictor.model_version}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
