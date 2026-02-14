"""
Drift Detection Module
Compares training data distributions with incoming data using KS-test and PSI.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "reference_data.csv"

# Thresholds
KS_P_VALUE_THRESHOLD = 0.05      # Reject null hypothesis if p < 0.05
PSI_THRESHOLD = 0.2              # PSI > 0.2 = significant drift
DRIFT_FEATURE_RATIO = 0.25       # Drift if >25% features drift

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def load_reference_data() -> pd.DataFrame:
    """Load the training reference data."""
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found: {REFERENCE_DATA_PATH}\n"
            "Run training first: python -m training.train"
        )
    return pd.read_csv(REFERENCE_DATA_PATH)


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI < 0.1  → No significant change
    PSI 0.1-0.2 → Moderate change
    PSI > 0.2  → Significant change
    
    Args:
        reference: Reference distribution
        current: Current distribution
        bins: Number of bins
    
    Returns:
        PSI value
    """
    # Create bins from reference data
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bin_edges = np.linspace(min_val - 1e-6, max_val + 1e-6, bins + 1)
    
    # Calculate proportions
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions (avoid zeros)
    ref_pct = (ref_counts + 1) / (len(reference) + bins)
    cur_pct = (cur_counts + 1) / (len(current) + bins)
    
    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return float(psi)


def ks_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None,
) -> Dict[str, Dict]:
    """
    Run Kolmogorov-Smirnov test per feature.
    
    Args:
        reference_df: Training data
        current_df: Incoming data
        features: Features to check
    
    Returns:
        Dict with per-feature results
    """
    features = features or FEATURE_NAMES
    results = {}
    
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        
        ref_values = reference_df[feature].dropna().values
        cur_values = current_df[feature].dropna().values
        
        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        
        results[feature] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < KS_P_VALUE_THRESHOLD,
        }
    
    return results


def psi_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None,
) -> Dict[str, Dict]:
    """
    Run PSI check per feature.
    
    Args:
        reference_df: Training data
        current_df: Incoming data
        features: Features to check
    
    Returns:
        Dict with per-feature PSI scores
    """
    features = features or FEATURE_NAMES
    results = {}
    
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        
        ref_values = reference_df[feature].dropna().values
        cur_values = current_df[feature].dropna().values
        
        psi_value = calculate_psi(ref_values, cur_values)
        
        results[feature] = {
            "psi": psi_value,
            "drift_detected": psi_value > PSI_THRESHOLD,
        }
    
    return results


def check_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame = None,
) -> Tuple[bool, Dict]:
    """
    Combined drift check using KS-test and PSI.
    
    Args:
        current_df: Incoming data
        reference_df: Training data (loaded from file if None)
    
    Returns:
        Tuple of (drift_detected: bool, report: dict)
    """
    if reference_df is None:
        reference_df = load_reference_data()
    
    # Run both tests
    ks_results = ks_drift_check(reference_df, current_df)
    psi_results = psi_drift_check(reference_df, current_df)
    
    # Count drifted features
    ks_drifted = [f for f, r in ks_results.items() if r["drift_detected"]]
    psi_drifted = [f for f, r in psi_results.items() if r["drift_detected"]]
    
    # Drift if either method detects enough drifted features
    total_features = len(FEATURE_NAMES)
    ks_ratio = len(ks_drifted) / total_features if total_features > 0 else 0
    psi_ratio = len(psi_drifted) / total_features if total_features > 0 else 0
    
    drift_detected = (ks_ratio >= DRIFT_FEATURE_RATIO) or (psi_ratio >= DRIFT_FEATURE_RATIO)
    
    report = {
        "drift_detected": drift_detected,
        "timestamp": datetime.now().isoformat(),
        "ks_test": {
            "drifted_features": ks_drifted,
            "drifted_count": len(ks_drifted),
            "total_features": total_features,
            "details": ks_results,
        },
        "psi": {
            "drifted_features": psi_drifted,
            "drifted_count": len(psi_drifted),
            "total_features": total_features,
            "details": psi_results,
        },
    }
    
    return drift_detected, report


def generate_report(report: Dict) -> str:
    """Format drift report as text."""
    lines = []
    lines.append("\n" + "=" * 50)
    lines.append("🔍 Drift Detection Report")
    lines.append("=" * 50)
    
    status = "🚨 DRIFT DETECTED" if report["drift_detected"] else "✅ NO DRIFT"
    lines.append(f"\nStatus: {status}")
    lines.append(f"Time: {report['timestamp'][:19]}")
    
    # KS-test results
    ks = report["ks_test"]
    lines.append(f"\n📊 KS-Test: {ks['drifted_count']}/{ks['total_features']} features drifted")
    for feature, detail in ks["details"].items():
        marker = "🚨" if detail["drift_detected"] else "✅"
        lines.append(f"   {marker} {feature}: stat={detail['statistic']:.4f}, p={detail['p_value']:.4f}")
    
    # PSI results
    psi = report["psi"]
    lines.append(f"\n📊 PSI: {psi['drifted_count']}/{psi['total_features']} features drifted")
    for feature, detail in psi["details"].items():
        marker = "🚨" if detail["drift_detected"] else "✅"
        lines.append(f"   {marker} {feature}: PSI={detail['psi']:.4f}")
    
    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def main():
    """CLI for drift detection."""
    parser = argparse.ArgumentParser(description="Drift Detection")
    parser.add_argument("--file", type=str, help="Path to incoming data CSV")
    parser.add_argument("--no-drift", action="store_true", help="Test with reference data (no drift expected)")
    args = parser.parse_args()
    
    if args.no_drift:
        # Test with same distribution — should detect NO drift
        print("🧪 Testing with reference data (expecting no drift)...")
        ref_df = load_reference_data()
        # Sample from same data
        current_df = ref_df.sample(n=min(1000, len(ref_df)), random_state=42)
        drift_detected, report = check_drift(current_df, ref_df)
    elif args.file:
        print(f"🧪 Checking drift for: {args.file}")
        current_df = pd.read_csv(args.file)
        drift_detected, report = check_drift(current_df)
    else:
        # Default: check for drifted data file
        drifted_path = PROJECT_ROOT / "data" / "drifted_data.csv"
        if drifted_path.exists():
            print(f"🧪 Checking drift for: {drifted_path}")
            current_df = pd.read_csv(drifted_path)
            drift_detected, report = check_drift(current_df)
        else:
            print("🧪 No data file specified. Testing with reference data...")
            ref_df = load_reference_data()
            current_df = ref_df.sample(n=min(1000, len(ref_df)), random_state=42)
            drift_detected, report = check_drift(current_df, ref_df)
    
    print(generate_report(report))
    return drift_detected


if __name__ == "__main__":
    main()
