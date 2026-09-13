# predcache

**Persistent timestamp-keyed prediction cache and batched inference runner for ML models.**

Extracted and generalized from a production neural-network trading system where
full-range backtests over 100,000+ bars were dominated by redundant model
inference. Storing predictions keyed by **bar timestamp** (not row index) makes
any sub-range replayable without offset alignment — and makes incremental
backtests reuse everything previously computed. The result: full-range runs
went from **hours to minutes**.

## Why

Typical ML backtesting pipelines recompute model inference for the same bars on
every run — every parameter sweep, every range extension, every "what if".
`predcache` decouples the two expensive parts:

1. **Feature-window assembly** — building look-back feature rows from an OHLC
   frame (most-recent-first, warmup-aware), with your domain logic plugged in
   via `row_fn`.
2. **Batched inference** — scoring rows through *any* model callable
   (Keras, PyTorch, ONNX, sklearn) in fixed batches with progress reporting.
3. **Persistent caching** — predictions keyed by timestamp, saved atomically,
   so the next run computes only what's missing.

## Install

```bash
pip install predcache              # core (numpy only)
pip install predcache[parquet]     # + parquet backend
```

## Quick start

```python
import pandas as pd
from predcache import PredictionCache, BatchedInferenceRunner, FeatureWindowAssembler

# 1. Assemble feature windows (your domain logic stays in row_fn)
assembler = FeatureWindowAssembler(
    window=50, warmup=500,
    row_fn=lambda win: my_feature_logic(win),   # 1-D feature vector per window
)
X, keys = assembler.build(df, key_col="time")   # X: (n_bars - warmup, n_features)

# 2. Batched inference through any model
runner = BatchedInferenceRunner(
    n_samples=len(X),
    inference_fn=lambda s, e: model.predict(X[s:e], verbose=0),
    batch_size=500,
)
preds = runner.run()

# 3. Cache by timestamp — next run computes only missing bars
cache = PredictionCache("predictions.pkl")
out = cache.get_or_compute(
    keys=keys,
    compute_fn=lambda missing: predict_bars(X, keys, missing),
)
```

## Features

- **Timestamp-keyed cache** — sub-range extraction without offset alignment;
  row-index caches silently break on slicing/resampling, timestamp keys don't.
- **Atomic saves** — temp file + `os.replace`; a crash mid-save never corrupts
  an existing cache.
- **Pluggable backends** — pickle (zero-dependency) and parquet ship in the
  box; register custom backends with `register_backend(name, dump, load)`.
- **Batched inference runner** — framework-agnostic `(start, stop) -> scores`
  contract, progress callbacks, tuned default batch size (500).
- **L-inf normalization** helper matching the source system's
  `normalize_vector_absolute`.

## Origin

This package is the generalized core of the prediction-serving layer of
[Neuromomentum](https://github.com/Geomanti)'s production trading system
(three Keras classifiers, MT5 live execution, 2012–2026 backtest range). The
original cache manager was 542 lines of domain-specific code; what ships here
is the reusable pattern: timestamp-keyed persistence, compute-only-missing,
batched scoring, windowed feature assembly.

## License

MIT — see [LICENSE](LICENSE).