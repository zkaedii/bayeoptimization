"""OpenTelemetry distributed tracing for Bayesian optimization services.

Configures a ``TracerProvider`` with an OTLP exporter whose endpoint is read
from the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.  When the
variable is absent the module falls back to no-op tracing so that missing
telemetry infrastructure never crashes application code.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode, Tracer
from opentelemetry.trace.propagation import get_current_span

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_LOCK = threading.RLock()


class OtelTracer:
    """OpenTelemetry distributed tracing wrapper.

    Automatically configures a ``TracerProvider`` backed by an OTLP gRPC
    exporter.  If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is not set, a no-op tracer
    is used instead — this guarantees that instrumentation never raises when
    telemetry infrastructure is unavailable.

    Params:
        service_name: Logical service name written into every span.
        service_version: Semantic version of the service.
        prime_phase: Current PRIME phase label attached to every span.
        otlp_endpoint: Override for the OTLP exporter endpoint.  When *None*
            the value is read from ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        An ``OtelTracer`` instance.

    Example::

        tracer = OtelTracer(
            service_name="bo-service",
            service_version="1.2.0",
            prime_phase="exploration",
        )

        @tracer.traced("my_function")
        def my_function(x: int) -> int:
            return x * 2
    """

    _instance: Optional[OtelTracer] = None
    _init_done: bool = False

    def __new__(
        cls,
        service_name: str = "bayeoptimization",
        service_version: str = "0.1.0",
        prime_phase: str = "unknown",
        otlp_endpoint: Optional[str] = None,
    ) -> OtelTracer:
        """Return the singleton ``OtelTracer`` instance.

        Params:
            service_name: Logical service name.
            service_version: Semantic version string.
            prime_phase: PRIME phase label.
            otlp_endpoint: Optional OTLP endpoint override.

        Returns:
            The singleton ``OtelTracer``.

        Example::

            a = OtelTracer()
            b = OtelTracer()
            assert a is b
        """
        if cls._instance is None:
            with _LOCK:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        service_name: str = "bayeoptimization",
        service_version: str = "0.1.0",
        prime_phase: str = "unknown",
        otlp_endpoint: Optional[str] = None,
    ) -> None:
        """Initialise the tracer provider (idempotent).

        Params:
            service_name: Logical service name.
            service_version: Semantic version string.
            prime_phase: PRIME phase label.
            otlp_endpoint: Optional OTLP endpoint override.

        Returns:
            None

        Example::

            tracer = OtelTracer(service_name="my-svc")
        """
        with _LOCK:
            if self.__class__._init_done:
                return
            self._init_locked(service_name, service_version, prime_phase, otlp_endpoint)

    def _init_locked(
        self,
        service_name: str,
        service_version: str,
        prime_phase: str,
        otlp_endpoint: Optional[str],
    ) -> None:
        """Perform actual initialisation under lock.

        Params:
            service_name: Logical service name.
            service_version: Semantic version string.
            prime_phase: PRIME phase label.
            otlp_endpoint: Optional OTLP endpoint override.

        Returns:
            None

        Example::

            # Called internally by __init__
        """
        self._service_name: str = service_name
        self._service_version: str = service_version
        self._prime_phase: str = prime_phase
        self._noop: bool = False

        endpoint: Optional[str] = otlp_endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )

        if endpoint is None:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT not set — using no-op tracer"
            )
            self._noop = True
            self._tracer: Tracer = trace.get_tracer(service_name, service_version)
        else:
            self._tracer = self._build_tracer(endpoint)

        self.__class__._init_done = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_tracer(self, endpoint: str) -> Tracer:
        """Create a fully-configured ``TracerProvider`` with OTLP export.

        Params:
            endpoint: OTLP gRPC endpoint URL.

        Returns:
            A configured ``Tracer`` instance.

        Example::

            tracer = self._build_tracer("http://localhost:4317")
        """
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc is not installed — "
                "falling back to no-op tracer"
            )
            self._noop = True
            return trace.get_tracer(self._service_name, self._service_version)

        resource = Resource.create(
            {
                "service.name": self._service_name,
                "service.version": self._service_version,
                "zkaedi.prime.phase": self._prime_phase,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info(
            "OtelTracer configured: endpoint=%s service=%s",
            endpoint,
            self._service_name,
        )
        return trace.get_tracer(self._service_name, self._service_version)

    def _default_attributes(self) -> Dict[str, str]:
        """Return span attributes that are attached to every span.

        Returns:
            Dict of default attribute key-value pairs.

        Example::

            attrs = self._default_attributes()
        """
        return {
            "service.name": self._service_name,
            "service.version": self._service_version,
            "zkaedi.prime.phase": self._prime_phase,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def traced(self, span_name: str) -> Callable[[F], F]:
        """Decorator that wraps a function in an OpenTelemetry span.

        On success the span ends normally; on exception the span status is set
        to ``ERROR`` and the exception is recorded before re-raising.

        Params:
            span_name: Human-readable name for the span.

        Returns:
            A decorator that instruments the target function.

        Example::

            tracer = OtelTracer()

            @tracer.traced("compute_acquisition")
            def compute_acquisition(x: float) -> float:
                return x ** 2
        """

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._tracer.start_as_current_span(
                    span_name, attributes=self._default_attributes()
                ) as span:
                    try:
                        result = fn(*args, **kwargs)
                        return result
                    except Exception as exc:
                        span.set_status(StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return cast(F, wrapper)

        return decorator

    def create_span(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Create and start a new span for manual instrumentation.

        The caller is responsible for ending the span via ``span.end()``.

        Params:
            name: Human-readable span name.
            attributes: Extra key-value attributes to attach.

        Returns:
            An active ``Span`` instance.

        Example::

            tracer = OtelTracer()
            span = tracer.create_span("custom_step", {"iteration": 5})
            try:
                do_work()
            finally:
                span.end()
        """
        merged: Dict[str, Any] = {**self._default_attributes()}
        if attributes:
            merged.update(attributes)
        span: Span = self._tracer.start_span(name, attributes=merged)
        return span

    # ------------------------------------------------------------------
    # Context propagation
    # ------------------------------------------------------------------
    @staticmethod
    def inject_context(headers: Dict[str, str]) -> Dict[str, str]:
        """Inject the current trace context into *headers* for propagation.

        Params:
            headers: Mutable mapping (e.g. HTTP headers) to inject into.

        Returns:
            The same *headers* dict, now containing trace-context entries.

        Example::

            headers: dict[str, str] = {}
            OtelTracer.inject_context(headers)
        """
        inject(headers)
        return headers

    @staticmethod
    def extract_context(headers: Dict[str, str]) -> Context:
        """Extract trace context from incoming *headers*.

        Params:
            headers: Mapping containing W3C Trace Context entries.

        Returns:
            An OpenTelemetry ``Context`` that can be attached.

        Example::

            ctx = OtelTracer.extract_context(request.headers)
        """
        return extract(headers)

    # ------------------------------------------------------------------
    # Phase update
    # ------------------------------------------------------------------
    def set_prime_phase(self, phase: str) -> None:
        """Update the PRIME phase label for future spans.

        Params:
            phase: New phase string (e.g. ``"exploitation"``).

        Returns:
            None

        Example::

            tracer = OtelTracer()
            tracer.set_prime_phase("exploitation")
        """
        self._prime_phase = phase
        logger.debug("PRIME phase updated to %s", phase)
