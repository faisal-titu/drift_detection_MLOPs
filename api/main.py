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
from .middleware import LatencyMiddleware, metrics_collector

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

# Observability middleware (latency tracking + SLO)
app.add_middleware(LatencyMiddleware)

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


@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """Return per-endpoint latency percentiles, error rates, and SLO breach counts."""
    return metrics_collector.snapshot()


@app.post("/rollback", tags=["Admin"])
async def rollback_model(target_version: int):
    """Roll back to a previous model version.

    Promotes the given version to production and reloads the serving model.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from registry.promote_model import promote_to_production, get_model_metadata

    metadata = get_model_metadata(target_version)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Model v{target_version} not found")

    success = promote_to_production(target_version, force=True)
    if not success:
        raise HTTPException(status_code=500, detail="Promotion failed")

    predictor.reload_model()
    logger.info("Rolled back to model v%d", target_version)
    return {
        "status": "rolled_back",
        "model_version": target_version,
        "r2": metadata.get("metrics", {}).get("r2"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
