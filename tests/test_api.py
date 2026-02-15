"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import predictor


@pytest.fixture(autouse=True)
def load_model():
    """Ensure model is loaded for tests."""
    if not predictor.is_loaded():
        predictor.load_model()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_valid_input(self, client):
        payload = {
            "MedInc": 5.0,
            "HouseAge": 30.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.0,
            "Population": 1000.0,
            "AveOccup": 3.0,
            "Latitude": 34.0,
            "Longitude": -118.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], float)

    def test_predict_missing_field(self, client):
        payload = {"MedInc": 5.0}  # Missing required fields
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

