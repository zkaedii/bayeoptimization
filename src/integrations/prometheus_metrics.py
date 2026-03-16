"""Prometheus metrics instrumentation for Bayesian optimization pipelines.

Provides pre-configured histograms, counters, and gauges for monitoring
optimization performance, drift detection, adversarial defense, and
resource usage.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry & lock
# ---------------------------------------------------------------------------
_REGISTRY: Optional[CollectorRegistry] = None
_LOCK = threading.Lock()


def _get_registry() -> CollectorRegistry:
    """Return the shared collector registry, creating it on first access.

    Returns:
        CollectorRegistry: The singleton Prometheus collector registry.

    Example::

        registry = _get_registry()
    """
    global _REGISTRY
    if _REGISTRY is None:
        with _LOCK:
            if _REGISTRY is None:
                _REGISTRY = CollectorRegistry(auto_describe=True)
    return _REGISTRY


class PrometheusMetrics:
    """Pre-configured Prometheus instrumentation for the optimization stack.

    All metrics are singletons — safe to import and instantiate from multiple
    modules.  Repeated instantiation returns the same underlying collectors.

    Params:
        registry: Optional custom ``CollectorRegistry``.  When *None* the
            module-level singleton registry is used.

    Returns:
        A ``PrometheusMetrics`` instance exposing histogram, counter, and
        gauge attributes.

    Example::

        metrics = PrometheusMetrics()
        metrics.optimization_latency_seconds.observe(0.42)
        metrics.bo_iterations_total.labels(
            optimizer="gp", acquisition="ei"
        ).inc()
        metrics.start_metrics_server(port=8001)
    """

    _instance: Optional[PrometheusMetrics] = None
    _init_done: bool = False

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    def __new__(cls, registry: Optional[CollectorRegistry] = None) -> PrometheusMetrics:
        """Ensure only one instance of PrometheusMetrics exists.

        Params:
            registry: Optional custom ``CollectorRegistry``.

        Returns:
            The singleton ``PrometheusMetrics`` instance.

        Example::

            a = PrometheusMetrics()
            b = PrometheusMetrics()
            assert a is b
        """
        if cls._instance is None:
            with _LOCK:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        """Initialise all Prometheus collectors (idempotent).

        Params:
            registry: Optional custom ``CollectorRegistry``.

        Returns:
            None

        Example::

            metrics = PrometheusMetrics()
        """
        if self.__class__._init_done:
            return
        self._registry: CollectorRegistry = registry or _get_registry()
        self._server_started: bool = False

        # ---- Histograms ------------------------------------------------
        self.optimization_latency_seconds: Histogram = Histogram(
            "optimization_latency_seconds",
            "End-to-end latency of a single optimisation step",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self._registry,
        )

        self.gp_training_time_seconds: Histogram = Histogram(
            "gp_training_time_seconds",
            "Time spent training the Gaussian-process surrogate",
            buckets=[0.001, 0.01, 0.1, 1.0],
            registry=self._registry,
        )

        self.inference_latency_seconds: Histogram = Histogram(
            "inference_latency_seconds",
            "Latency of a single model inference call",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self._registry,
        )

        # ---- Counters ---------------------------------------------------
        self.bo_iterations_total: Counter = Counter(
            "bo_iterations_total",
            "Total Bayesian-optimisation iterations executed",
            labelnames=["optimizer", "acquisition"],
            registry=self._registry,
        )

        self.drift_events_total: Counter = Counter(
            "drift_events_total",
            "Total drift events detected (DRIFT and CRITICAL)",
            labelnames=["phase"],
            registry=self._registry,
        )

        self.adversarial_attacks_total: Counter = Counter(
            "adversarial_attacks_total",
            "Total adversarial attacks observed",
            labelnames=["epsilon", "defended"],
            registry=self._registry,
        )

        self.unknown_class_rejections_total: Counter = Counter(
            "unknown_class_rejections_total",
            "Total unknown-class rejections by evidential classifier",
            registry=self._registry,
        )

        # ---- Gauges -----------------------------------------------------
        self.memory_usage_bytes: Gauge = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes per component",
            labelnames=["component"],
            registry=self._registry,
        )

        self.prime_field_variance: Gauge = Gauge(
            "prime_field_variance",
            "Current Hamiltonian field variance",
            labelnames=["phase"],
            registry=self._registry,
        )

        self.active_learning_uncertainty_mean: Gauge = Gauge(
            "active_learning_uncertainty_mean",
            "Mean uncertainty of the current active-learning pool",
            registry=self._registry,
        )

        self.__class__._init_done = True
        logger.info("PrometheusMetrics initialised with %d collectors", 10)

    # ------------------------------------------------------------------
    # Metrics server
    # ------------------------------------------------------------------
    def start_metrics_server(self, port: int = 8001) -> None:
        """Start the ``/metrics`` HTTP endpoint in a daemon thread.

        Calling this method more than once is a no-op; the server is started
        at most once per process lifetime.

        Params:
            port: TCP port for the metrics endpoint.

        Returns:
            None

        Example::

            metrics = PrometheusMetrics()
            metrics.start_metrics_server(port=9090)
        """
        if self._server_started:
            logger.debug("Metrics server already running — skipping start")
            return
        start_http_server(port, registry=self._registry)
        self._server_started = True
        logger.info("Prometheus metrics server started on port %d", port)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def record_drift_event(self, phase: str, severity: str) -> None:
        """Increment the drift counter for DRIFT and CRITICAL severities.

        Params:
            phase: Optimisation phase label (e.g. ``"exploration"``).
            severity: One of ``"DRIFT"`` or ``"CRITICAL"``.

        Returns:
            None

        Example::

            metrics = PrometheusMetrics()
            metrics.record_drift_event("exploration", "DRIFT")
        """
        if severity in ("DRIFT", "CRITICAL"):
            self.drift_events_total.labels(phase=phase).inc()
            logger.debug("Drift event recorded: phase=%s severity=%s", phase, severity)

    def update_memory(self, component: str, usage_bytes: float) -> None:
        """Set the current memory-usage gauge for *component*.

        Params:
            component: Logical component name (e.g. ``"gp_model"``).
            usage_bytes: Current memory consumption in bytes.

        Returns:
            None

        Example::

            metrics = PrometheusMetrics()
            metrics.update_memory("gp_model", 1_048_576)
        """
        self.memory_usage_bytes.labels(component=component).set(usage_bytes)
