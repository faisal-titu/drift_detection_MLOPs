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
from utils.logging_config import get_logger

logger = get_logger(__name__)


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
    Find the model version with the best R2 score.

    Returns:
        Tuple of (best_version, metadata) or (None, None) if no models
    """
    versions = get_all_versions()

    if not versions:
        logger.error("No model versions found")
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
        logger.info("Best model: v%d (R2 = %.4f)", best_version, best_r2)

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
        logger.error("Model v%d not found", version)
        return False

    if PRODUCTION_MODEL_DIR.exists() and not force:
        current_metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
        if current_metadata_path.exists():
            with open(current_metadata_path, "r") as f:
                current = json.load(f)
            logger.info(
                "Current production: v%s (R2 = %.4f)",
                current.get("version"),
                current["metrics"].get("r2", 0),
            )

    if PRODUCTION_MODEL_DIR.exists():
        shutil.rmtree(PRODUCTION_MODEL_DIR)

    shutil.copytree(source_dir, PRODUCTION_MODEL_DIR)

    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["promoted_to_production"] = True
    metadata["source_version"] = version

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model v%d promoted to production", version)
    logger.info("  Location: %s", PRODUCTION_MODEL_DIR)

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
    logger.info("=" * 50)
    logger.info("Model Registry - Promote to Production")
    logger.info("=" * 50)

    versions = get_all_versions()
    logger.info("Available versions: %s", versions)

    success = promote_best_model()

    if success:
        metadata = get_production_metadata()
        logger.info("=" * 50)
        logger.info("Production Model Ready")
        logger.info("  Version: v%s", metadata.get("source_version"))
        logger.info("  R2 Score: %.4f", metadata["metrics"].get("r2", 0))
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
