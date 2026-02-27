"""
Observability Middleware
- Request latency tracking (p50, p95, p99)
- Per-endpoint counters
- SLO breach alerting
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ── SLO Targets ──────────────────────────────────────────────────────
PREDICT_P95_SLO_MS = 150          # /predict p95 must stay below 150 ms
HEALTH_P95_SLO_MS = 50            # /health  p95 must stay below 50 ms
DEFAULT_P95_SLO_MS = 500          # catch-all

SLO_MAP = {
    "/predict": PREDICT_P95_SLO_MS,
    "/health": HEALTH_P95_SLO_MS,
}


@dataclass
class EndpointStats:
    """Accumulates latencies and counts for one endpoint."""
    latencies_ms: List[float] = field(default_factory=list)
    total_requests: int = 0
    error_count: int = 0
    slo_breaches: int = 0
    _max_samples: int = 10_000          # rolling window

    def record(self, latency_ms: float, status_code: int, slo_ms: float) -> None:
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        if status_code >= 500:
            self.error_count += 1
        if latency_ms > slo_ms:
            self.slo_breaches += 1
        # Trim to rolling window
        if len(self.latencies_ms) > self._max_samples:
            self.latencies_ms = self.latencies_ms[-self._max_samples:]

    def percentile(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, pct))

    def to_dict(self, slo_ms: float) -> Dict:
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(self.total_requests, 1), 4),
            "slo_target_ms": slo_ms,
            "slo_breaches": self.slo_breaches,
            "slo_breach_rate": round(self.slo_breaches / max(self.total_requests, 1), 4),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "avg_ms": round(float(np.mean(self.latencies_ms)) if self.latencies_ms else 0, 2),
            "max_ms": round(max(self.latencies_ms) if self.latencies_ms else 0, 2),
        }


class MetricsCollector:
    """Singleton that aggregates per-endpoint metrics."""

    def __init__(self) -> None:
        self._stats: Dict[str, EndpointStats] = defaultdict(EndpointStats)

    def record(self, path: str, latency_ms: float, status_code: int) -> None:
        slo_ms = SLO_MAP.get(path, DEFAULT_P95_SLO_MS)
        self._stats[path].record(latency_ms, status_code, slo_ms)

    def snapshot(self) -> Dict:
        """Return metrics summary for every tracked endpoint."""
        result: Dict = {}
        total_reqs = 0
        total_errs = 0
        all_latencies: List[float] = []
        for path, stats in sorted(self._stats.items()):
            slo_ms = SLO_MAP.get(path, DEFAULT_P95_SLO_MS)
            result[path] = stats.to_dict(slo_ms)
            total_reqs += stats.total_requests
            total_errs += stats.error_count
            all_latencies.extend(stats.latencies_ms)

        # Global summary
        result["_global"] = {
            "total_requests": total_reqs,
            "error_count": total_errs,
            "error_rate": round(total_errs / max(total_reqs, 1), 4),
            "p50_ms": round(float(np.percentile(all_latencies, 50)) if all_latencies else 0, 2),
            "p95_ms": round(float(np.percentile(all_latencies, 95)) if all_latencies else 0, 2),
            "p99_ms": round(float(np.percentile(all_latencies, 99)) if all_latencies else 0, 2),
        }
        return result

    def reset(self) -> None:
        self._stats.clear()


# Module-level singleton
metrics_collector = MetricsCollector()


class LatencyMiddleware(BaseHTTPMiddleware):
    """Measures request latency and feeds MetricsCollector."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        path = request.url.path
        metrics_collector.record(path, elapsed_ms, response.status_code)

        # Attach latency header for client visibility
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        # Log slow requests
        slo = SLO_MAP.get(path, DEFAULT_P95_SLO_MS)
        if elapsed_ms > slo:
            logger.warning(
                "SLO breach: %s %s took %.1f ms (target: %d ms)",
                request.method, path, elapsed_ms, slo,
            )

        return response
