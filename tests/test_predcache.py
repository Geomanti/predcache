import numpy as np
import pandas as pd
import pytest

from predcache import BatchedInferenceRunner, FeatureWindowAssembler, PredictionCache
from predcache.batched import stack_features
from predcache.windowing import normalize_absolute


# ---------------------------------------------------------------------------
# PredictionCache
# ---------------------------------------------------------------------------

def _fake_compute(keys):
    return {k: f"pred:{k}" for k in keys}


def test_cache_roundtrip(tmp_path):
    p = str(tmp_path / "cache.pkl")
    c = PredictionCache(p)
    out = c.get_or_compute(["a", "b", "c"], _fake_compute)
    assert out == {"a": "pred:a", "b": "pred:b", "c": "pred:c"}

    # fresh instance reads persisted state
    c2 = PredictionCache(p).load()
    assert "a" in c2 and len(c2) == 3
    assert c2.get("b") == "pred:b"


def test_cache_computes_only_missing(tmp_path):
    p = str(tmp_path / "cache.pkl")
    c = PredictionCache(p)
    calls = []

    def tracking_compute(keys):
        calls.append(list(keys))
        return _fake_compute(keys)

    c.get_or_compute(["a", "b"], tracking_compute)
    c.get_or_compute(["a", "b", "c"], tracking_compute)
    # second call must only ask for "c"
    assert calls == [["a", "b"], ["c"]]


def test_cache_atomic_save(tmp_path):
    p = str(tmp_path / "cache.pkl")
    c = PredictionCache(p)
    c.get_or_compute(["k1"], _fake_compute)
    # simulate leftover temp file — must not break loads
    (tmp_path / "cache.pkl.abc123.tmp").write_bytes(b"garbage")
    c2 = PredictionCache(p).load()
    assert c2.get("k1") == "pred:k1"


def test_cache_drop(tmp_path):
    p = str(tmp_path / "cache.pkl")
    c = PredictionCache(p)
    c.get_or_compute(["x", "y"], _fake_compute)
    assert c.drop(["x", "missing"]) == 1
    assert "x" not in c


def test_parquet_backend(tmp_path):
    pytest.importorskip("pyarrow")
    p = str(tmp_path / "cache.parquet")
    c = PredictionCache(p, backend="parquet")
    c.get_or_compute(["t1", "t2"], lambda ks: {k: [0.5, 1] for k in ks})
    c2 = PredictionCache(p, backend="parquet").load()
    assert c2.get("t1") == [0.5, 1]


# ---------------------------------------------------------------------------
# BatchedInferenceRunner
# ---------------------------------------------------------------------------

def test_runner_batches_and_scores():
    features = np.arange(1200 * 3, dtype=np.float32).reshape(1200, 3)
    seen = []

    def scorer(s, e):
        seen.append((s, e))
        return features[s:e].sum(axis=1)

    runner = BatchedInferenceRunner(len(features), scorer, batch_size=500)
    preds = runner.run()
    assert seen == [(0, 500), (500, 1000), (1000, 1200)]
    assert len(preds) == 1200
    # row 0 = 0+1+2 = 3; row 1199 sums the last 3 values of arange(3600)
    assert np.isclose(float(preds[0]), 3.0)
    assert np.isclose(float(preds[-1]), 3597.0 + 3598.0 + 3599.0)


def test_runner_progress():
    n = 1500
    runner = BatchedInferenceRunner(n, lambda s, e: np.ones(e - s), batch_size=500, progress_every=1)
    events = []
    runner.run(progress_cb=events.append)
    assert len(events) == 3
    assert events[-1]["rows_done"] == n
    assert runner.progress_log == events


def test_stack_features():
    m = stack_features([np.ones((2, 4)), np.zeros((3, 4))])
    assert m.shape == (5, 4) and m.dtype == np.float32


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _ohlc(n=30):
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min"),
            "close": np.linspace(1.0, 2.0, n),
            "high": np.linspace(1.1, 2.1, n),
            "low": np.linspace(0.9, 1.9, n),
        }
    )


def test_normalize_absolute():
    v = np.array([2.0, -1.0, 0.0], dtype=np.float32)
    out = normalize_absolute(v)
    assert np.isclose(out[0], 1.0) and np.isclose(out[1], -0.5)
    z = normalize_absolute(np.zeros(3))
    assert (z == 0).all()


def test_assembler_windows_and_keys():
    df = _ohlc(30)
    asm = FeatureWindowAssembler(window=5, warmup=10, row_fn=lambda w: [w["close"].iloc[0]])
    X, keys = asm.build(df, key_col="time")
    assert X.shape == (20, 1)
    assert len(keys) == 20
    assert keys[0] == df.iloc[10]["time"]
    # most-recent-first: window's first row is the bar at index 10
    assert np.isclose(X[0, 0], df.iloc[10]["close"])