"""
Auto-Retraining Pipeline
Orchestrates: drift detection -> retrain -> evaluate -> promote -> API reload
"""

import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import PRODUCTION_MODEL_DIR, get_latest_version
from training.train import retrain_with_data
from training.evaluate import is_model_acceptable
from training.preprocess import load_data
from drift.drift_check import check_drift, generate_report, load_reference_data
from drift.simulate_drift import generate_heavy_drift
from registry.promote_model import (
    promote_to_production,
    get_production_metadata,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

API_URL = "http://localhost:8000"


def get_current_production_r2() -> float:
    """Get R2 score of current production model."""
    metadata = get_production_metadata()
    if metadata and "metrics" in metadata:
        return metadata["metrics"].get("r2", 0.0)
    return 0.0


def reload_api_model() -> bool:
    """Trigger API model reload via /reload endpoint."""
    try:
        response = requests.post(f"{API_URL}/reload", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("API model reloaded: v%s", data.get("model_version"))
            return True
        else:
            logger.warning("API reload failed: %d", response.status_code)
            return False
    except requests.ConnectionError:
        logger.warning("API not running - reload skipped (model will load on next startup)")
        return False


def run_pipeline(
    incoming_data_path: Optional[str] = None,
    simulate: bool = False,
    force_retrain: bool = False,
    min_r2_improvement: float = 0.0,
) -> bool:
    """
    Run the full auto-retraining pipeline.

    Args:
        incoming_data_path: Path to incoming data CSV
        simulate: If True, simulate heavy drift data
        force_retrain: If True, skip drift check and retrain
        min_r2_improvement: Minimum R2 improvement required to promote new model

    Returns:
        True if retraining was triggered and successful
    """
    logger.info("=" * 60)
    logger.info("AUTO-RETRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Load incoming data ---
    logger.info("Step 1: Loading data...")

    if incoming_data_path:
        incoming_df = pd.read_csv(incoming_data_path)
        logger.info("  Loaded from: %s (%d samples)", incoming_data_path, len(incoming_df))
    elif simulate:
        logger.info("  Simulating drifted data...")
        reference_df = load_reference_data()
        incoming_df = generate_heavy_drift(reference_df, n_samples=1000)
        logger.info("  Generated %d drifted samples", len(incoming_df))
    else:
        default_path = PROJECT_ROOT / "data" / "drifted_data.csv"
        if default_path.exists():
            incoming_df = pd.read_csv(default_path)
            logger.info("  Loaded from: %s (%d samples)", default_path, len(incoming_df))
        else:
            logger.error("No incoming data found. Use --simulate or provide --data path")
            return False

    # --- Step 2: Drift check ---
    logger.info("Step 2: Checking for drift...")

    if force_retrain:
        logger.info("  Forced retrain - skipping drift check")
        drift_detected = True
    else:
        drift_detected, report = check_drift(incoming_df)

        if drift_detected:
            logger.info(generate_report(report))
        else:
            logger.info("  No drift detected. No retraining needed.")
            return False

    # --- Step 3: Retrain ---
    logger.info("Step 3: Retraining model...")

    reference_df = load_reference_data()
    feature_columns = reference_df.columns.drop("MedHouseVal")

    combined_X = pd.concat([
        reference_df.drop("MedHouseVal", axis=1),
        incoming_df[feature_columns]
    ], ignore_index=True)

    reference_y = reference_df["MedHouseVal"]

    if "MedHouseVal" in incoming_df.columns:
        extra_y = incoming_df["MedHouseVal"].reset_index(drop=True)
        logger.info("  Using labeled incoming data for retraining")
    else:
        import numpy as np
        logger.warning("  Incoming data has no target column 'MedHouseVal'. Falling back to sampled proxy labels")
        extra_y = pd.Series(
            np.random.choice(reference_y.values, size=len(incoming_df), replace=True),
            name="MedHouseVal"
        )

    combined_y = pd.concat([reference_y, extra_y], ignore_index=True)

    model, new_metrics, new_version = retrain_with_data(
        combined_X, combined_y, run_name=f"retrain_v{get_latest_version() + 1}"
    )

    # --- Step 4: Evaluate ---
    logger.info("Step 4: Evaluating new model...")

    current_r2 = get_current_production_r2()
    new_r2 = new_metrics["r2"]

    logger.info("  Current production R2: %.4f", current_r2)
    logger.info("  New model R2:          %.4f", new_r2)

    if not is_model_acceptable(new_metrics):
        logger.error("New model below minimum threshold. Keeping current model.")
        return False

    if current_r2 > 0 and new_r2 < (current_r2 + min_r2_improvement):
        logger.warning(
            "New model does not improve enough (required >= %.4f, got %.4f). Keeping current production model.",
            current_r2 + min_r2_improvement,
            new_r2,
        )
        return False

    # --- Step 5: Promote ---
    logger.info("Step 5: Promoting new model...")

    success = promote_to_production(new_version, force=True)
    if not success:
        logger.error("Promotion failed")
        return False

    # --- Step 6: Reload API ---
    logger.info("Step 6: Reloading API model...")
    reload_api_model()

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("AUTO-RETRAINING COMPLETE")
    logger.info("  Old model: R2 = %.4f", current_r2)
    logger.info("  New model: v%d, R2 = %.4f", new_version, new_r2)
    logger.info("=" * 60)

    return True


def main():
    """CLI for auto-retraining pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Retraining Pipeline")
    parser.add_argument("--data", type=str, help="Path to incoming data CSV")
    parser.add_argument("--simulate", action="store_true", help="Simulate drifted data")
    parser.add_argument("--force", action="store_true", help="Force retrain (skip drift check)")
    parser.add_argument(
        "--min-r2-improvement",
        type=float,
        default=0.0,
        help="Only promote when new R2 is at least current R2 + this margin",
    )
    args = parser.parse_args()

    success = run_pipeline(
        incoming_data_path=args.data,
        simulate=args.simulate,
        force_retrain=args.force,
        min_r2_improvement=args.min_r2_improvement,
    )

    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.info("Pipeline finished - no retraining performed")


if __name__ == "__main__":
    main()
