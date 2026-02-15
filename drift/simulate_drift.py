"""
Drift Simulation Module
Generates synthetic data with controlled amounts of drift for testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_no_drift(reference_df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Generate data with same distribution as reference (no drift)."""
    return reference_df.sample(n=min(n_samples, len(reference_df)),
                                replace=True, random_state=42).reset_index(drop=True)


def generate_mild_drift(reference_df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Generate data with mild distribution shift."""
    sampled = reference_df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)

    # Shift 2 features slightly
    sampled["MedInc"] = sampled["MedInc"] * 1.3 + 0.5
    sampled["HouseAge"] = sampled["HouseAge"] + 5

    return sampled


def generate_heavy_drift(reference_df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Generate data with heavy distribution shift."""
    sampled = reference_df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)

    # Shift most features significantly
    sampled["MedInc"] = sampled["MedInc"] * 2.0 + 3.0
    sampled["HouseAge"] = sampled["HouseAge"] * 0.5 + 20
    sampled["AveRooms"] = sampled["AveRooms"] * 1.5 + 2
    sampled["AveBedrms"] = sampled["AveBedrms"] * 1.8
    sampled["Population"] = sampled["Population"] * 0.3
    sampled["AveOccup"] = sampled["AveOccup"] * 2.0

    return sampled


def main():
    """Generate drift simulation datasets."""
    from drift.drift_check import load_reference_data

    logger.info("Generating drift simulation data...")

    reference_df = load_reference_data()
    logger.info("Reference data: %d samples", len(reference_df))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # No drift
    no_drift_df = generate_no_drift(reference_df)
    no_drift_path = DATA_DIR / "no_drift_data.csv"
    no_drift_df.to_csv(no_drift_path, index=False)
    logger.info("No-drift data saved: %s (%d samples)", no_drift_path, len(no_drift_df))

    # Mild drift
    mild_drift_df = generate_mild_drift(reference_df)
    mild_drift_path = DATA_DIR / "mild_drift_data.csv"
    mild_drift_df.to_csv(mild_drift_path, index=False)
    logger.info("Mild-drift data saved: %s (%d samples)", mild_drift_path, len(mild_drift_df))

    # Heavy drift
    heavy_drift_df = generate_heavy_drift(reference_df)
    heavy_drift_path = DATA_DIR / "drifted_data.csv"
    heavy_drift_df.to_csv(heavy_drift_path, index=False)
    logger.info("Heavy-drift data saved: %s (%d samples)", heavy_drift_path, len(heavy_drift_df))

    # Summary
    logger.info("Feature comparison (mean):")
    for col in ["MedInc", "HouseAge", "AveRooms", "Population"]:
        if col in reference_df.columns:
            logger.info(
                "  %s: ref=%.2f, no_drift=%.2f, mild=%.2f, heavy=%.2f",
                col,
                reference_df[col].mean(),
                no_drift_df[col].mean(),
                mild_drift_df[col].mean(),
                heavy_drift_df[col].mean(),
            )


if __name__ == "__main__":
    main()
