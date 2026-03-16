"""Integrations package — Prometheus, OpenTelemetry, and MLflow instrumentation.

Imports are lazy to avoid hard dependency on optional packages (mlflow, opentelemetry).

Example::

    from src.integrations.prometheus_metrics import PrometheusMetrics
    from src.integrations.otel_tracer import OtelTracer
    from src.integrations.mlflow_tracker import MLflowTracker
"""

__all__: list[str] = [
    "MLflowTracker",
    "OtelTracer",
    "PrometheusMetrics",
]
