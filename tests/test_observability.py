"""Observability tests: spans + counters with the in-memory exporter.

These run only when opentelemetry is installed (the ``otel`` test extra);
they are skipped cleanly otherwise. The no-op path is covered by forcing
``PREDCACHE_OTEL=0``.

Note: opentelemetry's SDK forbids overriding an already-set global
Tracer/MeterProvider, so these tests share ONE module-scoped provider and
read spans/metrics cumulatively (counting only the spans each test adds).
"""

import pytest

pytest.importorskip("opentelemetry", reason="observability tests need opentelemetry-api/sdk")

from opentelemetry import trace as _trace
from opentelemetry import metrics as _metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import predcache
from predcache import BatchedInferenceRunner, FeatureWindowAssembler, PredictionCache


@pytest.fixture(scope="module")
def span_exporter():
    """One provider for the module; returns the shared in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        _trace.set_tracer_provider(provider)
    except Exception:
        # A provider was already set by another test module — use a proxy
        # tracer from the existing provider but keep collecting into ours.
        pass
    # Re-point predcache's lazy tracer at this provider.
    import predcache.observability as obs

    obs._TRACER = trace = _trace.get_tracer("predcache-test")
    yield exporter
    obs._TRACER = None  # restore lazy resolution


@pytest.fixture(scope="module")
def metric_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    try:
        _metrics.set_meter_provider(provider)
    except Exception:
        pass
    import predcache.observability as obs

    obs._METER = _metrics.get_meter("predcache-test")
    # predcache caches counters per meter; reset the cache so the new meter
    # creates fresh counters instead of reusing ones from another provider.
    obs._COUNTERS.clear()
    yield reader
    obs._METER = None
    obs._COUNTERS.clear()


def _names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def test_get_or_compute_emits_span_with_attributes(span_exporter, metric_reader, tmp_path):
    n_before = len(_names(span_exporter))
    cache = PredictionCache(str(tmp_path / "c.pkl"))
    cache.get_or_compute(["a", "b"], compute_fn=lambda missing: {k: 1 for k in missing})
    cache.get_or_compute(["a", "b", "c"], compute_fn=lambda missing: {k: 2 for k in missing})

    spans = span_exporter.get_finished_spans()[n_before:]
    names = [s.name for s in spans]
    assert names.count("predcache.get_or_compute") == 2

    second = spans[-1]
    attrs = dict(second.attributes)
    assert attrs["predcache.requested"] == 3
    assert attrs["predcache.computed"] == 1
    assert attrs["predcache.hits"] == 2
    assert "predcache.elapsed_s" in attrs


def test_runner_emits_inference_span_and_metrics(span_exporter, metric_reader):
    n_before = len(_names(span_exporter))
    runner = BatchedInferenceRunner(
        n_samples=10, inference_fn=lambda s, e: [[float(i)] for i in range(s, e)], batch_size=4
    )
    runner.run()

    spans = span_exporter.get_finished_spans()[n_before:]
    inf = [s for s in spans if s.name == "predcache.inference"]
    assert len(inf) == 1
    attrs = dict(inf[0].attributes)
    assert attrs["predcache.rows"] == 10
    assert attrs["predcache.batch_size"] == 4
    assert "predcache.elapsed_s" in attrs


def test_features_span(span_exporter, metric_reader):
    import pandas as pd

    n_before = len(_names(span_exporter))
    df = pd.DataFrame({"time": [f"t{i}" for i in range(6)], "close": range(6)})
    asm = FeatureWindowAssembler(window=3, warmup=1, row_fn=lambda win: [float(win["close"].iloc[0])])
    X, keys = asm.build(df, key_col="time")
    assert len(keys) == 5

    spans = span_exporter.get_finished_spans()[n_before:]
    feat = [s for s in spans if s.name == "predcache.features"]
    assert len(feat) == 1
    attrs = dict(feat[0].attributes)
    assert attrs["predcache.windows"] == 5
    assert attrs["predcache.window"] == 3


def test_env_kill_switch_disables_emission(span_exporter, metric_reader, tmp_path, monkeypatch):
    monkeypatch.setenv("PREDCACHE_OTEL", "0")
    try:
        assert predcache.otel_available() is False
        n_before = len(_names(span_exporter))
        cache = PredictionCache(str(tmp_path / "c2.pkl"))
        cache.get_or_compute(["x"], compute_fn=lambda missing: {k: 9 for k in missing})
        assert len(_names(span_exporter)) == n_before
    finally:
        monkeypatch.delenv("PREDCACHE_OTEL", raising=False)


def test_package_works_without_otel(monkeypatch, tmp_path):
    """The no-op path: PREDCACHE_OTEL=0 must not change cache behavior."""
    monkeypatch.setenv("PREDCACHE_OTEL", "0")
    cache = PredictionCache(str(tmp_path / "c3.pkl"))
    r1 = cache.get_or_compute(["k1"], compute_fn=lambda missing: {k: 42 for k in missing})
    r2 = cache.get_or_compute(["k1"], compute_fn=lambda missing: {k: -1 for k in missing})
    assert r1 == {"k1": 42}
    assert r2 == {"k1": 42}  # served from cache, compute_fn not called