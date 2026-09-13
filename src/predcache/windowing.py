"""Feature-window assembly helpers.

Generalizes the windowing pattern from the source trading system: for each
bar at index *i*, build a feature vector from a look-back window ending at
*i*, optionally reversed (most-recent-first, the convention used when the
models were trained on MT5 series ordering), then normalize.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .observability import _TimedSpan

__all__ = ["normalize_absolute", "FeatureWindowAssembler"]


def normalize_absolute(vec: np.ndarray) -> np.ndarray:
    """Scale a vector so every element lies in [-1, 1] (L-inf normalization).

    Matches the source system's ``normalize_vector_absolute``: divide by the
    max absolute value, preserving sign and zeros.  Returns the input
    unchanged when its max abs value is 0.
    """
    vec = np.asarray(vec, dtype=np.float32)
    scale = np.max(np.abs(vec))
    if scale == 0:
        return vec
    return vec / scale


class FeatureWindowAssembler:
    """Build look-back feature windows from an OHLC frame.

    Parameters
    ----------
    window
        Look-back length in bars (``bars_to_fetch`` in the source system).
    warmup
        Number of initial bars to skip before emitting windows.
    row_fn
        Callable ``(window_df) -> 1-D array`` producing the feature vector
        for one window.  Receives the reversed (most-recent-first) window
        slice.  This keeps domain feature logic (fractal detection, price
        deltas, external regressors) in *your* code — the assembler only
        handles looping, warmup and stacking.

    Examples
    --------
    >>> assembler = FeatureWindowAssembler(window=50, warmup=10, row_fn=my_features)
    >>> X = assembler.build(df)          # (n_bars - warmup, n_features)
    >>> keys = assembler.keys(df)        # matching timestamp keys
    """

    def __init__(
        self,
        window: int,
        warmup: int = 0,
        row_fn: Optional[Callable] = None,
    ):
        if window <= 0:
            raise ValueError("window must be positive")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.window = int(window)
        self.warmup = int(warmup)
        self.row_fn = row_fn

    def windows(self, df) -> List:
        """Yield reversed (most-recent-first) window slices starting at warmup."""
        n = len(df)
        for i in range(self.warmup, n):
            lo = max(0, i - (self.window - 1))
            yield df.iloc[lo : i + 1].iloc[::-1].reset_index(drop=True)

    def keys(self, df, key_col: str = "time") -> List:
        """Timestamp keys for each window (stable identity for caching)."""
        return [df.iloc[i][key_col] for i in range(self.warmup, len(df))]

    def build(self, df, key_col: Optional[str] = None):
        """Stack feature rows for every window; returns (X, keys)."""
        if self.row_fn is None:
            raise ValueError("row_fn is required to build features")
        with _TimedSpan(
            "predcache.features",
            {
                "predcache.windows": max(0, len(df) - self.warmup),
                "predcache.window": self.window,
                "predcache.warmup": self.warmup,
            },
        ):
            rows: List[np.ndarray] = []
            for win in self.windows(df):
                rows.append(np.asarray(self.row_fn(win), dtype=np.float32))
            if not rows:
                return np.zeros((0, 0), dtype=np.float32), []
            keys = self.keys(df, key_col) if key_col else self.keys(df)
            return np.stack(rows, axis=0), keys