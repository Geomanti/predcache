"""Optional OpenTelemetry instrumentation for predcache.

Design
------
* **Zero hard dependency** — when ``opentelemetry`` is not installed every
  helper here degrades to a no-op context manager / counter sink, so the
  core cache stays dependency-light.
* **Auto-enabled** — as soon as ``opentelemetry-api`` is importable, spans
  and counters flow to whatever ``TracerProvider``/``MeterProvider`` the
  host application configured (the OpenTelemetry API returns no-op
  tracers/meters until a provider is set, so unconfigured hosts pay
  near-zero cost).
* **Explicit off switch** — set ``PREDCACHE_OTEL=0`` to disable emission
  entirely (useful in hot loops or test matrices without the package).

Emitted signals
---------------
Spans (as current spans, so they nest under the caller's trace):

* ``predcache.get_or_compute`` — attributes ``predcache.requested``,
  ``predcache.computed``, ``predcache.hits``
* ``predcache.inference`` — ``predcache.rows``, ``predcache.batches``,
  ``predcache.batch_size``, ``predcache.elapsed_s``
* ``predcache.features`` — ``predcache.windows``, ``predcache.window``,
  ``predcache.warmup``

Counters:

* ``predcache.predictions.computed`` / ``predcache.predictions.hits``
* ``predcache.inference.batches`` / ``predcache.inference.rows``
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

__all__ = ["span", "add_count", "otel_available"]

try:  # opentelemetry-api is optional
    from opentelemetry import metrics, trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in no-dependency installs
    _OTEL_AVAILABLE = False

_TRACER = None
_METER = None
_COUNTERS: Dict[str, Any] = {}


def otel_available() -> bool:
    """True when opentelemetry is importable and not disabled by env var."""
    return _OTEL_AVAILABLE and os.environ.get("PREDCACHE_OTEL", "1") != "0"


def _tracer():
    global _TRACER
    if _TRACER is None and _OTEL_AVAILABLE:
        # ProxyTracer: resolves the real provider lazily, so caching here is safe
        # even when the host sets the TracerProvider after importing predcache.
        _TRACER = trace.get_tracer("predcache")
    return _TRACER


def _meter():
    global _METER
    if _METER is None and _OTEL_AVAILABLE:
        _METER = metrics.get_meter("predcache")
    return _METER


def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a span as the current span; no-op context when OTel is absent."""
    tracer = _tracer() if otel_available() else None
    if tracer is None:
        return nullcontext()
    return tracer.start_as_current_span(name, attributes=attributes or {})


def add_count(name: str, value: int = 1, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Increment a monotonic counter; no-op when OTel is absent."""
    meter = _meter() if otel_available() else None
    if meter is None:
        return
    counter = _COUNTERS.get(name)
    if counter is None:
        counter = meter.create_counter(name)
        _COUNTERS[name] = counter
    if attributes:
        counter.add(value, attributes)
    else:
        counter.add(value)


class _TimedSpan:
    """Span helper that also reports elapsed seconds as a span attribute."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self._name = name
        self._attributes = dict(attributes or {})
        self._cm = span(name, self._attributes)
        self._t0: Optional[float] = None

    def __enter__(self):
        self._cm.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is not None:
            elapsed = round(time.perf_counter() - self._t0, 6)
            current = trace.get_current_span() if _OTEL_AVAILABLE else None
            if current is not None and current.is_recording():
                current.set_attribute("predcache.elapsed_s", elapsed)
        return self._cm.__exit__(exc_type, exc, tb)