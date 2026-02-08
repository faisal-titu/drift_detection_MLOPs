"""
Predictor Module - Model Loading and Inference
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
PRODUCTION_MODEL_DIR = MODELS_DIR / "production"


class Predictor:
    """Handles model loading and prediction."""
    
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Optional[Dict] = None
        self.model_version: Optional[int] = None
        self.feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        
    def load_model(self) -> bool:
        """
        Load the production model.
        
        Returns:
            True if successful
        """
        model_path = PRODUCTION_MODEL_DIR / "model.joblib"
        scaler_path = PRODUCTION_MODEL_DIR / "scaler.joblib"
        metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            self.model_version = self.metadata.get("source_version", 
                                                    self.metadata.get("version", 0))
            
            print(f"✅ Model loaded: v{self.model_version}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def reload_model(self) -> bool:
        """Reload the model (useful after promotion)."""
        return self.load_model()
    
    def predict(self, features: Dict) -> Tuple[float, int]:
        """
        Make a prediction.
        
        Args:
            features: Dictionary of feature names to values
        
        Returns:
            Tuple of (prediction, model_version)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert dict to array in correct order
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return float(prediction), self.model_version
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global predictor instance
predictor = Predictor()
