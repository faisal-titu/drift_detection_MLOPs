"""
Drift Simulation Module
Generates synthetic drifted data for testing drift detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DATA_PATH = DATA_DIR / "reference_data.csv"

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def generate_no_drift(reference_df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """Generate data with similar distribution (no drift)."""
    return reference_df[FEATURE_NAMES].sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)


def generate_mild_drift(reference_df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """
    Generate mildly drifted data.
    Shifts 2 features by ~1 standard deviation.
    """
    df = reference_df[FEATURE_NAMES].sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    
    # Shift MedInc up by 1 std
    df["MedInc"] = df["MedInc"] + reference_df["MedInc"].std()
    
    # Shift HouseAge down by 1 std
    df["HouseAge"] = df["HouseAge"] - reference_df["HouseAge"].std()
    
    return df


def generate_heavy_drift(reference_df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """
    Generate heavily drifted data.
    Shifts 5 features by 2+ standard deviations.
    """
    df = reference_df[FEATURE_NAMES].sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    
    # Shift multiple features significantly
    df["MedInc"] = df["MedInc"] + 2.5 * reference_df["MedInc"].std()
    df["HouseAge"] = df["HouseAge"] - 2.0 * reference_df["HouseAge"].std()
    df["AveRooms"] = df["AveRooms"] * 1.8
    df["Population"] = df["Population"] * 2.5
    df["AveOccup"] = df["AveOccup"] + 2.0 * reference_df["AveOccup"].std()
    
    # Clip negatives
    df = df.clip(lower=0)
    
    return df


def main():
    """Generate all drift scenarios."""
    print("🧪 Generating drift simulation data...\n")
    
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    print(f"📊 Reference data: {len(reference_df)} samples")
    
    # No drift
    no_drift_df = generate_no_drift(reference_df)
    no_drift_path = DATA_DIR / "no_drift_data.csv"
    no_drift_df.to_csv(no_drift_path, index=False)
    print(f"✅ No-drift data saved: {no_drift_path} ({len(no_drift_df)} samples)")
    
    # Mild drift
    mild_drift_df = generate_mild_drift(reference_df)
    mild_drift_path = DATA_DIR / "mild_drift_data.csv"
    mild_drift_df.to_csv(mild_drift_path, index=False)
    print(f"✅ Mild-drift data saved: {mild_drift_path} ({len(mild_drift_df)} samples)")
    
    # Heavy drift
    heavy_drift_df = generate_heavy_drift(reference_df)
    heavy_drift_path = DATA_DIR / "drifted_data.csv"
    heavy_drift_df.to_csv(heavy_drift_path, index=False)
    print(f"✅ Heavy-drift data saved: {heavy_drift_path} ({len(heavy_drift_df)} samples)")
    
    # Summary
    print("\n📊 Feature comparison (mean):")
    print(f"{'Feature':<12} {'Reference':>10} {'No Drift':>10} {'Mild':>10} {'Heavy':>10}")
    print("-" * 55)
    for f in FEATURE_NAMES:
        print(f"{f:<12} {reference_df[f].mean():>10.2f} {no_drift_df[f].mean():>10.2f} "
              f"{mild_drift_df[f].mean():>10.2f} {heavy_drift_df[f].mean():>10.2f}")


if __name__ == "__main__":
    main()
