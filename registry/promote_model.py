"""
Model Registry - Promote Model to Production
Finds the best performing model and promotes it to production.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from training.config import (
    MODELS_DIR,
    PRODUCTION_MODEL_DIR,
    get_latest_version,
)


def get_model_metadata(version: int) -> Optional[Dict]:
    """
    Load metadata for a specific model version.
    
    Args:
        version: Model version number
    
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = MODELS_DIR / f"v{version}" / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_all_versions() -> list:
    """
    Get all available model versions.
    
    Returns:
        List of version numbers
    """
    versions = []
    
    if not MODELS_DIR.exists():
        return versions
    
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            versions.append(int(d.name[1:]))
    
    return sorted(versions)


def get_best_model() -> Tuple[Optional[int], Optional[Dict]]:
    """
    Find the model version with the best R² score.
    
    Returns:
        Tuple of (best_version, metadata) or (None, None) if no models
    """
    versions = get_all_versions()
    
    if not versions:
        print("❌ No model versions found")
        return None, None
    
    best_version = None
    best_r2 = -float("inf")
    best_metadata = None
    
    for version in versions:
        metadata = get_model_metadata(version)
        if metadata and "metrics" in metadata:
            r2 = metadata["metrics"].get("r2", -float("inf"))
            if r2 > best_r2:
                best_r2 = r2
                best_version = version
                best_metadata = metadata
    
    if best_version:
        print(f"🏆 Best model: v{best_version} (R² = {best_r2:.4f})")
    
    return best_version, best_metadata


def promote_to_production(version: int, force: bool = False) -> bool:
    """
    Promote a model version to production.
    
    Args:
        version: Model version to promote
        force: Overwrite existing production model
    
    Returns:
        True if successful
    """
    source_dir = MODELS_DIR / f"v{version}"
    
    if not source_dir.exists():
        print(f"❌ Model v{version} not found")
        return False
    
    # Check current production model
    if PRODUCTION_MODEL_DIR.exists() and not force:
        current_metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
        if current_metadata_path.exists():
            with open(current_metadata_path, "r") as f:
                current = json.load(f)
            print(f"ℹ️ Current production: v{current.get('version')} (R² = {current['metrics'].get('r2', 'N/A'):.4f})")
    
    # Clear and copy
    if PRODUCTION_MODEL_DIR.exists():
        shutil.rmtree(PRODUCTION_MODEL_DIR)
    
    shutil.copytree(source_dir, PRODUCTION_MODEL_DIR)
    
    # Update metadata to mark as production
    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    metadata["promoted_to_production"] = True
    metadata["source_version"] = version
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model v{version} promoted to production!")
    print(f"   Location: {PRODUCTION_MODEL_DIR}")
    
    return True


def promote_best_model() -> bool:
    """
    Find and promote the best model to production.
    
    Returns:
        True if successful
    """
    best_version, metadata = get_best_model()
    
    if best_version is None:
        return False
    
    return promote_to_production(best_version, force=True)


def get_production_metadata() -> Optional[Dict]:
    """
    Get metadata for the current production model.
    
    Returns:
        Metadata dictionary or None
    """
    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def main():
    """CLI for model promotion."""
    print("\n" + "="*50)
    print("🏭 Model Registry - Promote to Production")
    print("="*50 + "\n")
    
    # Show all versions
    versions = get_all_versions()
    print(f"📦 Available versions: {versions}")
    print()
    
    # Find and promote best
    success = promote_best_model()
    
    if success:
        metadata = get_production_metadata()
        print("\n" + "="*50)
        print("🎉 Production Model Ready!")
        print(f"   Version: v{metadata.get('source_version')}")
        print(f"   R² Score: {metadata['metrics'].get('r2', 'N/A'):.4f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()
