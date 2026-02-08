"""
Pydantic Models for API Request/Response
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for housing prediction."""
    
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.024,
                    "Population": 322.0,
                    "AveOccup": 2.556,
                    "Latitude": 37.88,
                    "Longitude": -122.23
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for housing prediction."""
    
    prediction: float = Field(..., description="Predicted median house value (in $100,000s)")
    model_version: int = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction_id: int = Field(..., description="Unique prediction ID")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_version: Optional[int] = None
