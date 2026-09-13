"""Batched inference runner.

Wraps any model callable in batching + progress reporting, decoupling the
*scoring* of a feature matrix from the *loop* that feeds it.  In the source
production system this exact pattern (batch_size=500 across three Keras
classifiers) turned hours-long full-range backtests into minutes by making
the cost of inference independent of how many bars were requested.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .observability import _TimedSpan, add_count

__all__ = ["BatchedInferenceRunner", "InferenceFn"]

# A scorer takes (start, stop) and returns an array-like of predictions
# for that slice.  Returning (n_samples, n_outputs) is typical.
InferenceFn = Callable[[int, int], Any]

DEFAULT_BATCH_SIZE = 500


class BatchedInferenceRunner:
    """Run a scorer over *n_samples* rows in fixed batches.

    Parameters
    ----------
    n_samples
        Total number of feature rows to score.
    inference_fn
        Callable ``(start, stop) -> array-like`` that scores rows
        ``[start, stop)`` of the pre-built feature matrix.
    batch_size
        Rows per scoring call.  500 was tuned on the source system
        (Keras predict overhead vs memory footprint); any positive int works.
    progress_every
        Emit a progress dict every *progress_every* batches via the
        ``progress`` callback (or collect them in ``self.progress_log``).

    Examples
    --------
    >>> import numpy as np
    >>> features = np.random.rand(1000, 35).astype(np.float32)
    >>> runner = BatchedInferenceRunner(
    ...     n_samples=len(features),
    ...     inference_fn=lambda s, e: model.predict(features[s:e], verbose=0),
    ...     batch_size=500,
    ... )
    >>> preds = runner.run()
    """

    def __init__(
        self,
        n_samples: int,
        inference_fn: InferenceFn,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_every: int = 20,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.n_samples = int(n_samples)
        self.inference_fn = inference_fn
        self.batch_size = int(batch_size)
        self.progress_every = max(1, int(progress_every))
        self.progress_log: List[Dict[str, Any]] = []

    def batches(self) -> Iterator[tuple]:
        start = 0
        while start < self.n_samples:
            stop = min(start + self.batch_size, self.n_samples)
            yield start, stop
            start = stop

    def run(self, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Any]:
        """Score all rows; returns a list of per-row predictions."""
        out: List[Any] = []
        t0 = time.perf_counter()
        n_batches = 0
        with _TimedSpan(
            "predcache.inference",
            {
                "predcache.rows": self.n_samples,
                "predcache.batch_size": self.batch_size,
            },
        ):
            for start, stop in self.batches():
                chunk = self.inference_fn(start, stop)
                out.extend(_row_iter(chunk))
                n_batches += 1
                add_count("predcache.inference.batches", 1)
                add_count("predcache.inference.rows", stop - start)
                if progress_cb and n_batches % self.progress_every == 0:
                    cb = {
                        "rows_done": stop,
                        "rows_total": self.n_samples,
                        "batches": n_batches,
                        "elapsed_s": round(time.perf_counter() - t0, 3),
                    }
                    self.progress_log.append(cb)
                    progress_cb(cb)
        return out


def _row_iter(chunk: Any) -> Iterator[Any]:
    arr = np.asarray(chunk)
    if arr.ndim == 1:
        for v in arr:
            yield v.item() if hasattr(v, "item") else v
    else:
        for row in arr:
            yield row


def stack_features(matrices: Sequence[Any]) -> np.ndarray:
    """Concatenate per-bar feature rows into one matrix (float32).

    Utility used before calling a runner; keeps dtype consistent so
    framework predict() calls do not re-copy the array.
    """
    return np.concatenate([np.asarray(m, dtype=np.float32) for m in matrices], axis=0)