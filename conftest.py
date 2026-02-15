"""Pytest configuration — suppress third-party warnings."""
import warnings

# MLflow's PromptModelConfig triggers a Pydantic protected namespace warning
# during import. This is an upstream issue we cannot fix.
warnings.filterwarnings("ignore", message="Field.*model_.*protected namespace", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
