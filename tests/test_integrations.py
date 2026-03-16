"""Comprehensive tests for the integrations package.

Covers:
- PrometheusMetrics: singleton, histograms, counters, gauges, server, reset,
  get_metrics, custom metric creation.
- OtelTracer: traced decorator, create_span, inject/extract context, no-op
  behaviour (all mocked since opentelemetry is not installed).
- MLflowTracker: context manager, log_params, log_metrics, log_bo_step,
  log_prime_state, start_experiment, start_run, end_run (all mocked since
  mlflow is not installed).
"""

from __future__ import annotations

import importlib
import sys
import threading
from types import ModuleType, TracebackType
from typing import Any, Dict, Optional, Type
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from prometheus_client import CollectorRegistry, generate_latest

# ---------------------------------------------------------------------------
# PrometheusMetrics tests
# ---------------------------------------------------------------------------


def _fresh_prometheus_metrics(registry: Optional[CollectorRegistry] = None):
    """Import and return a *fresh* PrometheusMetrics class with singleton reset.

    Because PrometheusMetrics is a singleton with class-level state, every test
    must reset the class before instantiation to ensure isolation.
    """
    import src.integrations.prometheus_metrics as pm_mod

    # Reset singleton state
    pm_mod.PrometheusMetrics._instance = None
    pm_mod.PrometheusMetrics._init_done = False
    # Reset module-level registry so each test gets its own
    pm_mod._REGISTRY = None

    if registry is not None:
        return pm_mod.PrometheusMetrics(registry=registry)
    return pm_mod.PrometheusMetrics(registry=CollectorRegistry())


class TestPrometheusMetricsSingleton:
    """Verify the singleton pattern for PrometheusMetrics."""

    def setup_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False
        pm_mod._REGISTRY = None

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False
        pm_mod._REGISTRY = None

    def test_singleton_returns_same_instance(self) -> None:
        """Two instantiations must yield the same object."""
        import src.integrations.prometheus_metrics as pm_mod

        reg = CollectorRegistry()
        a = pm_mod.PrometheusMetrics(registry=reg)
        b = pm_mod.PrometheusMetrics(registry=reg)
        assert a is b

    def test_init_runs_only_once(self) -> None:
        """_init_done prevents re-registration of collectors."""
        import src.integrations.prometheus_metrics as pm_mod

        reg = CollectorRegistry()
        m1 = pm_mod.PrometheusMetrics(registry=reg)
        original_histogram = m1.optimization_latency_seconds
        m2 = pm_mod.PrometheusMetrics(registry=reg)
        assert m2.optimization_latency_seconds is original_histogram

    def test_default_registry_is_created_automatically(self) -> None:
        """When no registry is passed, the module creates one."""
        import src.integrations.prometheus_metrics as pm_mod

        m = pm_mod.PrometheusMetrics()
        assert m._registry is not None
        assert isinstance(m._registry, CollectorRegistry)


class TestPrometheusMetricsHistograms:
    """Verify that histogram observations are recorded correctly."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_optimization_latency_seconds_observe(self) -> None:
        self.metrics.optimization_latency_seconds.observe(0.42)
        output = generate_latest(self.registry).decode()
        assert "optimization_latency_seconds" in output
        assert 'optimization_latency_seconds_count 1.0' in output

    def test_gp_training_time_seconds_observe(self) -> None:
        self.metrics.gp_training_time_seconds.observe(0.005)
        self.metrics.gp_training_time_seconds.observe(0.015)
        output = generate_latest(self.registry).decode()
        assert 'gp_training_time_seconds_count 2.0' in output

    def test_inference_latency_seconds_observe(self) -> None:
        self.metrics.inference_latency_seconds.observe(0.003)
        output = generate_latest(self.registry).decode()
        assert 'inference_latency_seconds_count 1.0' in output

    def test_histogram_buckets_present(self) -> None:
        self.metrics.optimization_latency_seconds.observe(0.02)
        output = generate_latest(self.registry).decode()
        # Should contain the custom bucket boundaries
        assert 'optimization_latency_seconds_bucket{le="0.05"}' in output
        assert 'optimization_latency_seconds_bucket{le="5.0"}' in output


class TestPrometheusMetricsCounters:
    """Verify all counter increments."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_bo_iterations_total_increment(self) -> None:
        self.metrics.bo_iterations_total.labels(optimizer="gp", acquisition="ei").inc()
        self.metrics.bo_iterations_total.labels(optimizer="gp", acquisition="ei").inc()
        output = generate_latest(self.registry).decode()
        assert 'bo_iterations_total' in output
        assert 'optimizer="gp"' in output
        assert 'acquisition="ei"' in output

    def test_drift_events_total_increment(self) -> None:
        self.metrics.drift_events_total.labels(phase="exploration").inc()
        output = generate_latest(self.registry).decode()
        assert 'drift_events_total' in output
        assert 'phase="exploration"' in output

    def test_adversarial_attacks_total_increment(self) -> None:
        self.metrics.adversarial_attacks_total.labels(
            epsilon="0.1", defended="true"
        ).inc()
        output = generate_latest(self.registry).decode()
        assert 'adversarial_attacks_total' in output
        assert 'epsilon="0.1"' in output

    def test_unknown_class_rejections_total_increment(self) -> None:
        self.metrics.unknown_class_rejections_total.inc()
        self.metrics.unknown_class_rejections_total.inc()
        self.metrics.unknown_class_rejections_total.inc()
        output = generate_latest(self.registry).decode()
        assert "unknown_class_rejections_total 3.0" in output


class TestPrometheusMetricsGauges:
    """Verify all gauge set operations."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_memory_usage_bytes_set(self) -> None:
        self.metrics.memory_usage_bytes.labels(component="gp_model").set(1_048_576)
        output = generate_latest(self.registry).decode()
        assert "memory_usage_bytes" in output
        assert 'component="gp_model"' in output
        # Value may render as scientific notation (1.048576e+06)
        assert "1.048576e+06" in output or "1048576.0" in output

    def test_prime_field_variance_set(self) -> None:
        self.metrics.prime_field_variance.labels(phase="exploitation").set(0.42)
        output = generate_latest(self.registry).decode()
        assert "prime_field_variance" in output
        assert 'phase="exploitation"' in output

    def test_active_learning_uncertainty_mean_set(self) -> None:
        self.metrics.active_learning_uncertainty_mean.set(0.88)
        output = generate_latest(self.registry).decode()
        assert "active_learning_uncertainty_mean 0.88" in output

    def test_gauge_can_increase_and_decrease(self) -> None:
        gauge = self.metrics.active_learning_uncertainty_mean
        gauge.set(5.0)
        gauge.inc(2.0)
        gauge.dec(1.0)
        # Should be 6.0
        output = generate_latest(self.registry).decode()
        assert "active_learning_uncertainty_mean 6.0" in output


class TestPrometheusMetricsConvenienceHelpers:
    """Verify record_drift_event and update_memory helpers."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_record_drift_event_drift_severity(self) -> None:
        self.metrics.record_drift_event("exploration", "DRIFT")
        output = generate_latest(self.registry).decode()
        assert 'drift_events_total' in output
        assert 'phase="exploration"' in output

    def test_record_drift_event_critical_severity(self) -> None:
        self.metrics.record_drift_event("exploitation", "CRITICAL")
        output = generate_latest(self.registry).decode()
        assert 'phase="exploitation"' in output

    def test_record_drift_event_ignores_low_severity(self) -> None:
        """Severities other than DRIFT/CRITICAL should not increment."""
        self.metrics.record_drift_event("exploration", "WARNING")
        output = generate_latest(self.registry).decode()
        # Counter family is registered but no child with phase="exploration"
        # should have been created
        assert 'phase="exploration"' not in output

    def test_update_memory(self) -> None:
        self.metrics.update_memory("gp_model", 2_097_152)
        output = generate_latest(self.registry).decode()
        assert 'component="gp_model"' in output
        # Value may render as scientific notation (2.097152e+06)
        assert "2.097152e+06" in output or "2097152.0" in output


class TestPrometheusMetricsServer:
    """Verify start/stop of the metrics HTTP server."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    @patch("src.integrations.prometheus_metrics.start_http_server")
    def test_start_metrics_server(self, mock_start: MagicMock) -> None:
        self.metrics.start_metrics_server(port=9090)
        mock_start.assert_called_once_with(9090, registry=self.registry)
        assert self.metrics._server_started is True

    @patch("src.integrations.prometheus_metrics.start_http_server")
    def test_start_metrics_server_idempotent(self, mock_start: MagicMock) -> None:
        """Calling start twice should only invoke start_http_server once."""
        self.metrics.start_metrics_server(port=9090)
        self.metrics.start_metrics_server(port=9090)
        mock_start.assert_called_once()

    @patch("src.integrations.prometheus_metrics.start_http_server")
    def test_start_metrics_server_default_port(self, mock_start: MagicMock) -> None:
        self.metrics.start_metrics_server()
        mock_start.assert_called_once_with(8001, registry=self.registry)


class TestPrometheusGetMetrics:
    """Verify that generate_latest produces expected output."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_get_metrics_output_contains_all_families(self) -> None:
        """All registered metric families should appear in the output."""
        output = generate_latest(self.registry).decode()
        expected_families = [
            "optimization_latency_seconds",
            "gp_training_time_seconds",
            "inference_latency_seconds",
            "bo_iterations_total",
            "drift_events_total",
            "adversarial_attacks_total",
            "unknown_class_rejections_total",
            "memory_usage_bytes",
            "prime_field_variance",
            "active_learning_uncertainty_mean",
        ]
        for family in expected_families:
            assert family in output, f"Missing metric family: {family}"

    def test_get_metrics_output_after_observations(self) -> None:
        self.metrics.optimization_latency_seconds.observe(1.5)
        self.metrics.unknown_class_rejections_total.inc()
        self.metrics.active_learning_uncertainty_mean.set(0.77)
        output = generate_latest(self.registry).decode()
        assert "optimization_latency_seconds_count 1.0" in output
        assert "unknown_class_rejections_total 1.0" in output
        assert "active_learning_uncertainty_mean 0.77" in output


class TestPrometheusReset:
    """Verify that resetting the singleton allows fresh re-initialization."""

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_reset_allows_new_registry(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        reg1 = CollectorRegistry()
        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False
        m1 = pm_mod.PrometheusMetrics(registry=reg1)

        # Reset singleton
        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

        reg2 = CollectorRegistry()
        m2 = pm_mod.PrometheusMetrics(registry=reg2)

        assert m1 is not m2
        assert m1._registry is not m2._registry

    def test_reset_clears_server_started_flag(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        reg = CollectorRegistry()
        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False
        m = pm_mod.PrometheusMetrics(registry=reg)
        m._server_started = True

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False
        m2 = pm_mod.PrometheusMetrics(registry=CollectorRegistry())
        assert m2._server_started is False


class TestPrometheusCustomMetric:
    """Verify that custom metrics can be registered on the shared registry."""

    def setup_method(self) -> None:
        self.registry = CollectorRegistry()
        self.metrics = _fresh_prometheus_metrics(self.registry)

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_custom_counter_on_shared_registry(self) -> None:
        from prometheus_client import Counter

        custom = Counter(
            "custom_events_total",
            "A custom counter for testing",
            registry=self.registry,
        )
        custom.inc(10)
        output = generate_latest(self.registry).decode()
        assert "custom_events_total 10.0" in output

    def test_custom_gauge_on_shared_registry(self) -> None:
        from prometheus_client import Gauge

        custom = Gauge(
            "custom_temperature",
            "A custom gauge for testing",
            registry=self.registry,
        )
        custom.set(98.6)
        output = generate_latest(self.registry).decode()
        assert "custom_temperature 98.6" in output

    def test_custom_histogram_on_shared_registry(self) -> None:
        from prometheus_client import Histogram

        custom = Histogram(
            "custom_duration_seconds",
            "A custom histogram for testing",
            buckets=[0.1, 0.5, 1.0],
            registry=self.registry,
        )
        custom.observe(0.3)
        output = generate_latest(self.registry).decode()
        assert "custom_duration_seconds_count 1.0" in output
        assert 'custom_duration_seconds_bucket{le="0.5"} 1.0' in output


class TestGetRegistry:
    """Verify the module-level _get_registry helper."""

    def teardown_method(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod._REGISTRY = None
        pm_mod.PrometheusMetrics._instance = None
        pm_mod.PrometheusMetrics._init_done = False

    def test_get_registry_creates_singleton(self) -> None:
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod._REGISTRY = None
        r1 = pm_mod._get_registry()
        r2 = pm_mod._get_registry()
        assert r1 is r2
        assert isinstance(r1, CollectorRegistry)

    def test_get_registry_thread_safe(self) -> None:
        """Concurrent calls must return the same registry."""
        import src.integrations.prometheus_metrics as pm_mod

        pm_mod._REGISTRY = None
        results: list = []

        def get() -> None:
            results.append(pm_mod._get_registry())

        threads = [threading.Thread(target=get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# OtelTracer tests (fully mocked -- opentelemetry not installed)
# ---------------------------------------------------------------------------

def _build_otel_mocks():
    """Build a complete mock environment for OtelTracer import.

    Returns a dict of mock modules and key mock objects needed for assertions.
    """
    # Root modules
    mock_otel = MagicMock()
    mock_trace = MagicMock()
    mock_context_mod = MagicMock()
    mock_propagate = MagicMock()
    mock_sdk = MagicMock()
    mock_sdk_resources = MagicMock()
    mock_sdk_trace = MagicMock()
    mock_sdk_trace_export = MagicMock()
    mock_trace_propagation = MagicMock()

    # Simulated StatusCode enum
    class _StatusCode:
        ERROR = "ERROR"
        OK = "OK"
    mock_trace.StatusCode = _StatusCode

    # Simulated trace.get_tracer
    mock_tracer_instance = MagicMock()
    mock_trace.get_tracer.return_value = mock_tracer_instance

    # Span mock returned by start_as_current_span context manager
    mock_span = MagicMock()
    mock_tracer_instance.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=mock_span
    )
    mock_tracer_instance.start_as_current_span.return_value.__exit__ = MagicMock(
        return_value=False
    )

    # start_span for create_span
    mock_manual_span = MagicMock()
    mock_tracer_instance.start_span.return_value = mock_manual_span

    # inject / extract
    mock_propagate.inject = MagicMock()
    mock_propagate.extract = MagicMock(return_value=MagicMock())

    modules = {
        "opentelemetry": mock_otel,
        "opentelemetry.context": mock_context_mod,
        "opentelemetry.trace": mock_trace,
        "opentelemetry.propagate": mock_propagate,
        "opentelemetry.sdk": mock_sdk,
        "opentelemetry.sdk.resources": mock_sdk_resources,
        "opentelemetry.sdk.trace": mock_sdk_trace,
        "opentelemetry.sdk.trace.export": mock_sdk_trace_export,
        "opentelemetry.trace.propagation": mock_trace_propagation,
    }

    # Assign sub-attributes so attribute access works
    mock_otel.context = mock_context_mod
    mock_otel.trace = mock_trace
    mock_otel.propagate = mock_propagate

    return modules, {
        "trace": mock_trace,
        "tracer_instance": mock_tracer_instance,
        "span": mock_span,
        "manual_span": mock_manual_span,
        "propagate": mock_propagate,
    }


def _import_otel_tracer(modules_dict: dict):
    """Force-import otel_tracer with the given mock modules in sys.modules."""
    # Remove cached module to ensure fresh import
    sys.modules.pop("src.integrations.otel_tracer", None)
    with mock.patch.dict(sys.modules, modules_dict):
        mod = importlib.import_module("src.integrations.otel_tracer")
        # Reset singleton for test isolation
        mod.OtelTracer._instance = None
        mod.OtelTracer._init_done = False
        return mod


class TestOtelTracerTracedDecorator:
    """Verify the @traced decorator creates spans and handles errors."""

    def test_traced_wraps_function_in_span(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)

        tracer = mod.OtelTracer(service_name="test-svc")

        @tracer.traced("my_func")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10
        mocks["tracer_instance"].start_as_current_span.assert_called()
        call_args = mocks["tracer_instance"].start_as_current_span.call_args
        assert call_args[0][0] == "my_func"

    def test_traced_records_exception(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(service_name="test-svc")

        @tracer.traced("failing_func")
        def failing_func() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_func()

        # The span should have set_status and record_exception called
        mocks["span"].set_status.assert_called()
        mocks["span"].record_exception.assert_called()

    def test_traced_preserves_function_name(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(service_name="test-svc")

        @tracer.traced("original_name")
        def original_name() -> str:
            return "ok"

        assert original_name.__name__ == "original_name"


class TestOtelTracerCreateSpan:
    """Verify create_span returns a properly configured span."""

    def test_create_span_basic(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(service_name="test-svc")

        span = tracer.create_span("my_span")
        mocks["tracer_instance"].start_span.assert_called_once()
        call_kwargs = mocks["tracer_instance"].start_span.call_args
        assert call_kwargs[0][0] == "my_span"
        assert span is mocks["manual_span"]

    def test_create_span_with_attributes(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(service_name="test-svc")

        extra = {"iteration": 5, "custom_key": "val"}
        tracer.create_span("step", attributes=extra)
        call_kwargs = mocks["tracer_instance"].start_span.call_args
        merged_attrs = call_kwargs[1]["attributes"]
        assert merged_attrs["iteration"] == 5
        assert merged_attrs["custom_key"] == "val"
        assert merged_attrs["service.name"] == "test-svc"

    def test_create_span_includes_default_attributes(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(
            service_name="svc", service_version="2.0", prime_phase="explore"
        )

        tracer.create_span("s")
        call_kwargs = mocks["tracer_instance"].start_span.call_args
        attrs = call_kwargs[1]["attributes"]
        assert attrs["service.name"] == "svc"
        assert attrs["service.version"] == "2.0"
        assert attrs["zkaedi.prime.phase"] == "explore"


class TestOtelTracerContextPropagation:
    """Verify inject_context and extract_context."""

    def test_inject_context(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        # No need to instantiate for static methods, but do for import coverage
        mod.OtelTracer(service_name="svc")

        headers: Dict[str, str] = {"existing": "value"}
        result = mod.OtelTracer.inject_context(headers)
        mocks["propagate"].inject.assert_called_once_with(headers)
        assert result is headers

    def test_extract_context(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        mod.OtelTracer(service_name="svc")

        headers = {"traceparent": "00-abc-def-01"}
        ctx = mod.OtelTracer.extract_context(headers)
        mocks["propagate"].extract.assert_called_once_with(headers)
        assert ctx is not None


class TestOtelTracerNoOp:
    """Verify no-op behaviour when OTEL_EXPORTER_OTLP_ENDPOINT is unset."""

    def test_noop_when_no_endpoint(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)

        with mock.patch.dict("os.environ", {}, clear=True):
            # Ensure OTEL_EXPORTER_OTLP_ENDPOINT is NOT set
            import os
            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            tracer = mod.OtelTracer(service_name="svc")

        assert tracer._noop is True
        # get_tracer should have been called (no-op path)
        mocks["trace"].get_tracer.assert_called()

    def test_noop_traced_still_executes_function(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)

        with mock.patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            tracer = mod.OtelTracer(service_name="svc")

        @tracer.traced("noop_func")
        def noop_func(x: int) -> int:
            return x + 1

        assert noop_func(10) == 11


class TestOtelTracerSetPrimePhase:
    """Verify that set_prime_phase updates the phase attribute."""

    def test_set_prime_phase(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        tracer = mod.OtelTracer(service_name="svc", prime_phase="init")
        assert tracer._prime_phase == "init"
        tracer.set_prime_phase("exploitation")
        assert tracer._prime_phase == "exploitation"


class TestOtelTracerSingleton:
    """Verify the singleton pattern for OtelTracer."""

    def test_singleton_returns_same_instance(self) -> None:
        modules, mocks = _build_otel_mocks()
        mod = _import_otel_tracer(modules)
        a = mod.OtelTracer(service_name="svc")
        b = mod.OtelTracer(service_name="svc")
        assert a is b


# ---------------------------------------------------------------------------
# MLflowTracker tests (fully mocked -- mlflow not installed)
# ---------------------------------------------------------------------------

def _build_mlflow_mocks():
    """Build a complete mock environment for MLflowTracker import."""
    mock_mlflow = MagicMock()
    mock_mlflow_tracking = MagicMock()
    mock_client_instance = MagicMock()
    mock_mlflow_tracking.MlflowClient.return_value = mock_client_instance

    # Mock active run object
    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "run-abc-123"
    mock_mlflow.start_run.return_value = mock_active_run

    # Mock experiment lookup
    mock_mlflow.get_experiment_by_name.return_value = None
    mock_mlflow.create_experiment.return_value = "exp-001"

    modules = {
        "mlflow": mock_mlflow,
        "mlflow.tracking": mock_mlflow_tracking,
    }
    return modules, {
        "mlflow": mock_mlflow,
        "client": mock_client_instance,
        "active_run": mock_active_run,
    }


def _import_mlflow_tracker(modules_dict: dict):
    """Force-import mlflow_tracker with mock modules."""
    sys.modules.pop("src.integrations.mlflow_tracker", None)
    with mock.patch.dict(sys.modules, modules_dict):
        return importlib.import_module("src.integrations.mlflow_tracker")


class TestMLflowTrackerContextManager:
    """Verify context manager enter/exit and automatic run cleanup."""

    def test_context_manager_returns_self(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker(tracking_uri="http://localhost:5000")
        with tracker as t:
            assert t is tracker

    def test_context_manager_ends_run_on_normal_exit(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        tracker._run_id = "some-run"
        with tracker:
            pass
        mocks["mlflow"].end_run.assert_called_once_with(status="FINISHED")

    def test_context_manager_ends_run_as_failed_on_exception(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        tracker._run_id = "some-run"
        with pytest.raises(RuntimeError):
            with tracker:
                raise RuntimeError("kaboom")
        mocks["mlflow"].end_run.assert_called_once_with(status="FAILED")

    def test_context_manager_no_run_does_not_call_end(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        # _run_id is None by default
        with tracker:
            pass
        mocks["mlflow"].end_run.assert_not_called()


class TestMLflowTrackerLogParams:
    """Verify log_params delegates correctly."""

    def test_log_params(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        params = {"learning_rate": 0.01, "n_iter": 100}
        tracker.log_params(params)
        mocks["mlflow"].log_params.assert_called_once_with(params)


class TestMLflowTrackerLogMetrics:
    """Verify log_metrics delegates with step."""

    def test_log_metrics(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        metrics = {"rmse": 0.12, "r2": 0.95}
        tracker.log_metrics(metrics, step=10)
        mocks["mlflow"].log_metrics.assert_called_once_with(metrics, step=10)


class TestMLflowTrackerLogBoStep:
    """Verify log_bo_step logs metrics and candidate param."""

    def test_log_bo_step(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        tracker.log_bo_step(
            step=5,
            x_candidate=[0.3, 0.7],
            y_observed=1.23,
            acquisition_value=0.45,
            gp_hyperparams={
                "length_scale": 0.5,
                "amplitude": 1.0,
                "noise_alpha": 0.01,
            },
        )

        # log_metrics should be called with the BO metrics
        metrics_call = mocks["mlflow"].log_metrics.call_args
        logged_metrics = metrics_call[0][0]
        assert logged_metrics["y_observed"] == 1.23
        assert logged_metrics["acquisition_value"] == 0.45
        assert logged_metrics["gp_length_scale"] == 0.5
        assert logged_metrics["gp_amplitude"] == 1.0
        assert logged_metrics["gp_noise_alpha"] == 0.01
        assert metrics_call[1]["step"] == 5

        # log_param should be called with x_candidate
        mocks["mlflow"].log_param.assert_called_once_with(
            "x_candidate_step_5", str([0.3, 0.7])
        )

    def test_log_bo_step_missing_hyperparams(self) -> None:
        """Missing GP hyperparams should default to 0.0."""
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        tracker.log_bo_step(
            step=0,
            x_candidate=[1.0],
            y_observed=0.5,
            acquisition_value=0.1,
            gp_hyperparams={},
        )

        logged_metrics = mocks["mlflow"].log_metrics.call_args[0][0]
        assert logged_metrics["gp_length_scale"] == 0.0
        assert logged_metrics["gp_amplitude"] == 0.0
        assert logged_metrics["gp_noise_alpha"] == 0.0


class TestMLflowTrackerLogPrimeState:
    """Verify log_prime_state logs PRIME metrics and phase tag."""

    def test_log_prime_state(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        tracker.log_prime_state(
            step=10,
            phase="exploration",
            variance=0.42,
            eta=0.1,
            gamma=0.9,
            beta=1.5,
        )

        logged_metrics = mocks["mlflow"].log_metrics.call_args[0][0]
        assert logged_metrics["prime_variance"] == 0.42
        assert logged_metrics["prime_eta"] == 0.1
        assert logged_metrics["prime_gamma"] == 0.9
        assert logged_metrics["prime_beta"] == 1.5
        assert mocks["mlflow"].log_metrics.call_args[1]["step"] == 10

        mocks["mlflow"].set_tag.assert_called_once_with("prime_phase", "exploration")


class TestMLflowTrackerExperimentLifecycle:
    """Verify start_experiment, start_run, end_run."""

    def test_start_experiment_creates_new(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        # Experiment does not exist
        mocks["mlflow"].get_experiment_by_name.return_value = None
        mocks["mlflow"].create_experiment.return_value = "exp-new-001"

        exp_id = tracker.start_experiment("new_experiment")
        assert exp_id == "exp-new-001"
        mocks["mlflow"].create_experiment.assert_called_once_with(
            "new_experiment", artifact_location=None
        )
        mocks["mlflow"].set_experiment.assert_called_once_with("new_experiment")

    def test_start_experiment_retrieves_existing(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-existing-42"
        mocks["mlflow"].get_experiment_by_name.return_value = mock_exp

        exp_id = tracker.start_experiment("existing_exp")
        assert exp_id == "exp-existing-42"
        mocks["mlflow"].create_experiment.assert_not_called()

    def test_start_run(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        tracker._experiment_id = "exp-001"

        run_id = tracker.start_run("run-42", tags={"variant": "A"})
        assert run_id == "run-abc-123"
        mocks["mlflow"].start_run.assert_called_once_with(
            run_name="run-42",
            experiment_id="exp-001",
            tags={"variant": "A"},
        )
        assert tracker._run_id == "run-abc-123"

    def test_end_run_finished(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        tracker._run_id = "some-run-id"

        tracker.end_run(status="FINISHED")
        mocks["mlflow"].end_run.assert_called_once_with(status="FINISHED")
        assert tracker._run_id is None

    def test_end_run_no_active_run(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        # _run_id is None
        tracker.end_run()
        mocks["mlflow"].end_run.assert_not_called()

    def test_end_run_failed(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        tracker._run_id = "run-xyz"

        tracker.end_run(status="FAILED")
        mocks["mlflow"].end_run.assert_called_once_with(status="FAILED")
        assert tracker._run_id is None


class TestMLflowTrackerInit:
    """Verify initialization behaviour."""

    def test_init_sets_tracking_uri(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        mod.MLflowTracker(tracking_uri="http://mlflow:5000")
        mocks["mlflow"].set_tracking_uri.assert_called_once_with(
            "http://mlflow:5000"
        )

    def test_init_no_tracking_uri(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        mod.MLflowTracker()
        mocks["mlflow"].set_tracking_uri.assert_not_called()

    def test_init_stores_artifact_location(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker(artifact_location="s3://bucket/artifacts")
        assert tracker._artifact_location == "s3://bucket/artifacts"

    def test_init_creates_client(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()
        assert tracker._client is mocks["client"]


class TestMLflowTrackerLogModel:
    """Verify log_model dispatches to the correct mlflow flavour."""

    def test_log_sklearn_model(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        # Create an object whose type().__module__ contains "sklearn"
        SklearnModel = type("MockGP", (), {})
        SklearnModel.__module__ = "sklearn.gaussian_process"
        mock_model = SklearnModel()

        tracker.log_model(mock_model, "gp_surrogate")
        mocks["mlflow"].sklearn.log_model.assert_called_once_with(
            mock_model, "gp_surrogate"
        )

    def test_log_torch_model(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        TorchModel = type("MockNet", (), {})
        TorchModel.__module__ = "torch.nn.modules"
        mock_model = TorchModel()

        tracker.log_model(mock_model, "neural_net")
        mocks["mlflow"].pytorch.log_model.assert_called_once_with(
            mock_model, "neural_net"
        )

    def test_log_generic_model_falls_back_to_pyfunc(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker()

        CustomModel = type("CustomModel", (), {})
        CustomModel.__module__ = "mypackage.models"
        mock_model = CustomModel()

        tracker.log_model(mock_model, "custom_model")
        mocks["mlflow"].pyfunc.log_model.assert_called_once_with(
            artifact_path="custom_model",
            python_model=mock_model,
        )


class TestMLflowTrackerStartExperimentWithArtifactLocation:
    """Verify artifact_location is passed through on experiment creation."""

    def test_artifact_location_passed_to_create(self) -> None:
        modules, mocks = _build_mlflow_mocks()
        mod = _import_mlflow_tracker(modules)
        tracker = mod.MLflowTracker(artifact_location="s3://my-bucket/art")

        mocks["mlflow"].get_experiment_by_name.return_value = None
        mocks["mlflow"].create_experiment.return_value = "exp-99"

        tracker.start_experiment("with_artifacts")
        mocks["mlflow"].create_experiment.assert_called_once_with(
            "with_artifacts", artifact_location="s3://my-bucket/art"
        )
