"""Pytest configuration - suppress third-party warnings."""
import warnings

# Suppress all common third-party warnings
warnings.filterwarnings("ignore", message="Field.*model_.*protected namespace", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
