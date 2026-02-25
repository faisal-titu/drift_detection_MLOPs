"""
Synthetic Streaming Drift Simulator
Streams labeled synthetic batches with changing feature-target relationships,
checks drift/performance, and triggers retraining when required.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from api.predictor import predictor
from drift.drift_check import FEATURE_NAMES, check_drift, load_reference_data
from training.evaluate import evaluate_model
from training.retrain_pipeline import run_pipeline
from utils.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
STREAM_DIR = PROJECT_ROOT / "data" / "stream_batches"

# ---- resource limits ----
_DEFAULT_MAX_RETRAINS = 3          # hard cap on retrains per simulation
_MIN_SLEEP_SECONDS   = 0.5        # enforced minimum pause between batches
_RETRAIN_N_JOBS      = 2          # cap RF parallelism during retrain


def _compute_target(
    df: pd.DataFrame, drift_intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Compute synthetic target.

    drift_intensity in [0, 1].  0 = original relationship, 1 = fully drifted.
    Values in between produce a smooth blend so the model degrades gradually.
    """
    med_inc = df["MedInc"].values
    house_age = df["HouseAge"].values
    ave_rooms = df["AveRooms"].values
    ave_bedrms = df["AveBedrms"].values
    population = df["Population"].values
    ave_occup = df["AveOccup"].values
    latitude = df["Latitude"].values
    longitude = df["Longitude"].values

    # --- stable relationship (what the model was trained on) ---
    target_stable = (
        0.50 * med_inc
        + 0.018 * house_age
        + 0.10 * ave_rooms
        - 0.30 * ave_bedrms
        - 0.00015 * population
        - 0.05 * ave_occup
        + 0.05 * (latitude - 36.0)
        - 0.025 * (longitude + 120.0)
        + rng.normal(0.0, 0.18, size=len(df))
    )

    if drift_intensity <= 0.0:
        return np.clip(target_stable, 0.2, 5.0)

    # --- drifted relationship ---
    target_drift = (
        0.25 * med_inc
        - 0.010 * house_age
        + 0.15 * ave_rooms
        - 0.10 * ave_bedrms
        - 0.00004 * population
        + 0.08 * ave_occup
        - 0.06 * (latitude - 36.0)
        + 0.04 * (longitude + 120.0)
        + 0.05 * med_inc * np.log1p(np.maximum(ave_rooms, 0.1))
        + rng.normal(0.0, 0.22, size=len(df))
        + 0.5
    )

    # smooth blend — drift_intensity=0.3 means 70% stable + 30% drifted
    target = (1.0 - drift_intensity) * target_stable + drift_intensity * target_drift
    return np.clip(target, 0.2, 5.0)


def _drift_intensity(batch_index: int, drift_start_batch: int, ramp_batches: int = 8) -> float:
    """Progressive drift intensity that ramps 0 → 1 over *ramp_batches* after drift_start.

    Returns 0.0 for stable batches and clamped at 1.0 for fully-drifted batches.
    """
    if batch_index < drift_start_batch:
        return 0.0
    progress = (batch_index - drift_start_batch) / max(ramp_batches, 1)
    return min(progress, 1.0)


def generate_stream_batch(
    reference_df: pd.DataFrame,
    batch_size: int,
    batch_index: int,
    drift_start_batch: int,
    rng: np.random.Generator,
    ramp_batches: int = 8,
) -> pd.DataFrame:
    """Generate one synthetic batch with **gradual** covariate + concept drift.

    Drift intensity ramps linearly from 0% at drift_start_batch to 100% over
    ramp_batches.  This avoids the scenario where every single feature flips
    to extreme values on one batch, allowing a realistic transition.
    """
    sampled = reference_df.sample(
        n=batch_size, replace=True,
        random_state=int(rng.integers(0, 1_000_000)),
    ).reset_index(drop=True)

    intensity = _drift_intensity(batch_index, drift_start_batch, ramp_batches)

    if intensity > 0.0:
        # Gentle covariate shifts scaled by intensity
        # Max multipliers are modest: ~1.15x instead of the old 1.8x
        sampled["MedInc"]     = sampled["MedInc"]     * (1.0 + intensity * rng.normal(0.12, 0.04, size=batch_size)) + intensity * 0.3
        sampled["HouseAge"]   = sampled["HouseAge"]   * (1.0 + intensity * rng.normal(-0.08, 0.03, size=batch_size)) + intensity * 2.0
        sampled["AveRooms"]   = sampled["AveRooms"]   * (1.0 + intensity * rng.normal(0.08, 0.03, size=batch_size)) + intensity * 0.3
        sampled["AveBedrms"]  = sampled["AveBedrms"]  * (1.0 + intensity * rng.normal(0.06, 0.02, size=batch_size))
        sampled["Population"] = sampled["Population"] * (1.0 + intensity * rng.normal(-0.10, 0.03, size=batch_size))
        sampled["AveOccup"]   = sampled["AveOccup"]   * (1.0 + intensity * rng.normal(0.10, 0.03, size=batch_size))
        sampled["Latitude"]   = sampled["Latitude"]   + intensity * rng.normal(-0.08, 0.03, size=batch_size)
        sampled["Longitude"]  = sampled["Longitude"]  + intensity * rng.normal(0.06, 0.02, size=batch_size)

    sampled["MedHouseVal"] = _compute_target(sampled, drift_intensity=intensity, rng=rng)
    return sampled


def evaluate_production_model(batch_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate current production model on a labeled batch."""
    if not predictor.is_loaded() and not predictor.load_model():
        raise RuntimeError("Production model is not available. Train and promote a model first.")

    X_batch = batch_df[FEATURE_NAMES]
    y_batch = batch_df["MedHouseVal"].values

    X_scaled = predictor.scaler.transform(X_batch)
    y_pred = predictor.model.predict(X_scaled)

    return evaluate_model(y_batch, y_pred)


def stream_and_monitor(
    num_batches: int = 20,
    batch_size: int = 300,
    drift_start_batch: int = 8,
    performance_r2_threshold: float = 0.40,
    min_r2_improvement: float = 0.02,
    sleep_seconds: float = 1.0,
    retrain_cooldown_batches: int = 5,
    random_state: int = 42,
    max_retrains: int = _DEFAULT_MAX_RETRAINS,
    ramp_batches: int = 8,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    stop_event: Optional["object"] = None,
) -> Tuple[int, int]:
    """Stream synthetic data and trigger retraining on drift/performance degradation.

    Stability features:
    - Drift intensity ramps gradually over *ramp_batches* after drift_start.
    - At most *max_retrains* retrains per simulation run.
    - Minimum sleep of ``_MIN_SLEEP_SECONDS`` is enforced between batches.
    - Retraining uses capped ``n_jobs`` to avoid saturating CPU.

    Returns:
        Tuple of (drift_or_perf_alerts, successful_retrains)
    """
    effective_sleep = max(sleep_seconds, _MIN_SLEEP_SECONDS)

    logger.info("=" * 72)
    logger.info("SYNTHETIC STREAMING DRIFT SIMULATION")
    logger.info(
        "batches=%d | batch_size=%d | total_samples=%d | drift_start=%d | ramp=%d",
        num_batches, batch_size, num_batches * batch_size, drift_start_batch, ramp_batches,
    )
    logger.info(
        "sleep=%.1fs | cooldown=%d | max_retrains=%d | retrain_n_jobs=%d",
        effective_sleep, retrain_cooldown_batches, max_retrains, _RETRAIN_N_JOBS,
    )
    logger.info("=" * 72)

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    # Cap retraining parallelism to protect CPU
    _prev_n_jobs = os.environ.get("DRIFT_RETRAIN_N_JOBS")
    os.environ["DRIFT_RETRAIN_N_JOBS"] = str(_RETRAIN_N_JOBS)

    rng = np.random.default_rng(random_state)
    reference_df = load_reference_data()

    alerts = 0
    retrains = 0
    last_retrain_batch = -10_000

    try:
        for batch_index in range(num_batches):
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                logger.info("Streaming stopped by user at batch %03d", batch_index)
                if progress_callback:
                    progress_callback({
                        "event": "stopped",
                        "batch_index": batch_index,
                        "alerts": alerts,
                        "retrains": retrains,
                    })
                break

            intensity = _drift_intensity(batch_index, drift_start_batch, ramp_batches)

            batch_df = generate_stream_batch(
                reference_df=reference_df,
                batch_size=batch_size,
                batch_index=batch_index,
                drift_start_batch=drift_start_batch,
                rng=rng,
                ramp_batches=ramp_batches,
            )

            batch_path = STREAM_DIR / f"batch_{batch_index:03d}.csv"
            batch_df.to_csv(batch_path, index=False)

            current_features = batch_df[FEATURE_NAMES]
            drift_detected, report = check_drift(current_features, reference_df)

            metrics = evaluate_production_model(batch_df)
            perf_drift = metrics["r2"] < performance_r2_threshold

            logger.info(
                "Batch %03d | intensity=%.0f%% | r2=%.4f | rmse=%.4f | drift=%s | perf_alert=%s",
                batch_index,
                intensity * 100,
                metrics["r2"],
                metrics["rmse"],
                drift_detected,
                perf_drift,
            )

            if progress_callback:
                progress_callback({
                    "event": "batch",
                    "batch_index": batch_index,
                    "num_batches": num_batches,
                    "intensity": round(intensity, 2),
                    "phase": "drifted" if intensity > 0 else "stable",
                    "r2": round(float(metrics["r2"]), 4),
                    "rmse": round(float(metrics["rmse"]), 4),
                    "drift_detected": bool(drift_detected),
                    "perf_alert": bool(perf_drift),
                    "alerts": alerts,
                    "retrains": retrains,
                })

            if drift_detected or perf_drift:
                alerts += 1

                retrain_allowed = (
                    (batch_index - last_retrain_batch) >= retrain_cooldown_batches
                    and retrains < max_retrains
                )

                if not retrain_allowed:
                    reason = "max_retrains_reached" if retrains >= max_retrains else "cooldown"
                    logger.info("  Retrain skipped (%s)", reason)
                    if progress_callback:
                        progress_callback({
                            "event": "retrain_skipped",
                            "batch_index": batch_index,
                            "reason": reason,
                            "cooldown": retrain_cooldown_batches,
                            "retrains": retrains,
                            "max_retrains": max_retrains,
                        })
                else:
                    logger.info("  Triggering retrain pipeline from batch %03d", batch_index)
                    success = run_pipeline(
                        incoming_data_path=str(batch_path),
                        force_retrain=perf_drift and not drift_detected,
                        min_r2_improvement=min_r2_improvement,
                    )
                    if success:
                        retrains += 1
                        last_retrain_batch = batch_index
                        predictor.reload_model()
                        reference_df = load_reference_data()

                    if progress_callback:
                        progress_callback({
                            "event": "retrain",
                            "batch_index": batch_index,
                            "success": bool(success),
                            "alerts": alerts,
                            "retrains": retrains,
                        })

                if drift_detected:
                    logger.info(
                        "  Drift summary | ks=%d/%d | psi=%d/%d",
                        report["ks_test"]["drifted_count"],
                        report["ks_test"]["total_features"],
                        report["psi"]["drifted_count"],
                        report["psi"]["total_features"],
                    )

            # Always sleep to avoid CPU saturation
            time.sleep(effective_sleep)
    finally:
        # Restore original env var
        if _prev_n_jobs is None:
            os.environ.pop("DRIFT_RETRAIN_N_JOBS", None)
        else:
            os.environ["DRIFT_RETRAIN_N_JOBS"] = _prev_n_jobs

    logger.info("=" * 72)
    logger.info("STREAM COMPLETE | alerts=%d | retrains=%d", alerts, retrains)
    logger.info("Saved streamed batches to: %s", STREAM_DIR)
    logger.info("=" * 72)

    if progress_callback:
        progress_callback({
            "event": "summary",
            "alerts": alerts,
            "retrains": retrains,
            "stream_dir": str(STREAM_DIR),
        })

    return alerts, retrains


def main() -> None:
    """CLI for synthetic streaming drift simulation."""
    parser = argparse.ArgumentParser(description="Synthetic Streaming Drift Simulation")
    parser.add_argument("--batches", type=int, default=20, help="Number of stream batches")
    parser.add_argument("--batch-size", type=int, default=300, help="Samples per batch")
    parser.add_argument("--drift-start", type=int, default=8, help="Batch index where drift starts")
    parser.add_argument("--perf-r2-threshold", type=float, default=0.40,
                        help="Trigger performance alert when production R2 is below this")
    parser.add_argument("--min-r2-improvement", type=float, default=0.02,
                        help="Only deploy new model if it beats current R2 by at least this margin")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Minimum batch gap between retrains")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Sleep seconds between batches (min 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-retrains", type=int, default=_DEFAULT_MAX_RETRAINS,
                        help="Max retrain attempts per simulation")
    parser.add_argument("--ramp-batches", type=int, default=8,
                        help="Batches over which drift ramps from 0%% to 100%%")

    args = parser.parse_args()

    stream_and_monitor(
        num_batches=args.batches,
        batch_size=args.batch_size,
        drift_start_batch=args.drift_start,
        performance_r2_threshold=args.perf_r2_threshold,
        min_r2_improvement=args.min_r2_improvement,
        sleep_seconds=args.sleep,
        retrain_cooldown_batches=args.cooldown,
        random_state=args.seed,
        max_retrains=args.max_retrains,
        ramp_batches=args.ramp_batches,
    )


if __name__ == "__main__":
    main()
