# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-13

### Added

- **OpenTelemetry instrumentation** (optional): spans
  `predcache.get_or_compute` / `predcache.inference` / `predcache.features`
  with request/compute/hit/rows/batch-size/elapsed attributes, plus counters
  `predcache.predictions.computed` / `.hits` and
  `predcache.inference.batches` / `.rows`.
- `otel` extra (`pip install predcache[otel]`); opentelemetry-api remains
  optional — without it the package behaves identically (no-op spans).
- `PREDCACHE_OTEL=0` environment switch to disable emission entirely.
- `otel_available()` public helper.

## [0.1.0] - 2026-09-13

### Added

- `PredictionCache`: persistent, timestamp-keyed prediction cache with
  atomic saves, compute-only-missing semantics (`get_or_compute`), key
  dropping, and pluggable backends (`pickle`, `parquet`, custom via
  `register_backend`).
- `BatchedInferenceRunner`: framework-agnostic batched scoring with
  configurable batch size (default 500, tuned in the source production
  system), progress callbacks, and `stack_features` utility.
- `FeatureWindowAssembler`: look-back window construction from OHLC frames
  (most-recent-first convention, warmup handling, stable timestamp keys)
  with user-supplied feature logic (`row_fn`).
- `normalize_absolute`: L-inf normalization matching the source system's
  `normalize_vector_absolute`.
- Test suite (pytest), GitHub Actions CI, packaging metadata.

### Notes

- Generalized from the prediction-serving layer of a production
  neural-network trading system (timestamp-keyed cache + 500-bar batched
  inference across three Keras classifiers).