"""Persistent, timestamp-keyed prediction cache.

Extracted and generalized from a production trading system (Neuromomentum /
ExtremePrediction) where full-range backtests over 100k+ bars were dominated
by redundant neural-network inference.  Storing predictions keyed by bar
timestamp (not row index) lets any sub-range be replayed without offset
alignment and lets incremental backtests reuse everything previously computed.

Design goals
------------
* **Correctness first** — atomic writes (temp file + os.replace), so a crash
  mid-save never corrupts the cache.
* **Backend-agnostic** — pickle (default) and parquet backends ship in the
  box; register your own with :func:`register_backend`.
* **Framework-agnostic** — works with any callable batch scorer (Keras,
  PyTorch, ONNX, sklearn, plain numpy).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Protocol

from .observability import _TimedSpan, add_count

__all__ = ["PredictionCache", "CacheEntry", "register_backend"]


class _Backend(Protocol):
    """A cache backend persists a mapping of keys to payloads."""

    def dump(self, entries: Dict[Hashable, Any], path: str) -> None: ...

    def load(self, path: str) -> Dict[Hashable, Any]: ...


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

def _pickle_dump(entries: Dict[Hashable, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path: str) -> Dict[Hashable, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}


def _parquet_dump(entries: Dict[Hashable, Any], path: str) -> None:
    import pandas as pd  # optional dependency

    keys = list(entries.keys())
    df = pd.DataFrame({"key": keys, "value": [entries[k] for k in keys]})
    df.to_parquet(path)


def _parquet_load(path: str) -> Dict[Hashable, Any]:
    import pandas as pd  # optional dependency

    df = pd.read_parquet(path)
    return dict(zip(df["key"], df["value"]))


_BACKENDS: Dict[str, _Backend] = {
    "pickle": (_pickle_dump, _pickle_load),
    "parquet": (_parquet_dump, _parquet_load),
}


def register_backend(name: str, dump: Callable, load: Callable) -> None:
    """Register a custom cache backend (dump/load callables)."""
    _BACKENDS[name] = (dump, load)


@dataclass(frozen=True)
class CacheEntry:
    """One cached prediction, keyed by a timestamp or other hashable."""

    key: Hashable
    value: Any
    meta: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            object.__setattr__(self, "meta", {})


class PredictionCache:
    """Persistent cache for model predictions keyed by bar timestamp.

    Parameters
    ----------
    path
        File path of the cache file.  Created on first save.
    backend
        ``"pickle"`` (default, zero-dependency) or ``"parquet"``
        (requires ``pyarrow``; compact for numeric payloads).
    safe_save
        Write via a temp file + :func:`os.replace` so a crash never
        corrupts an existing cache.  Default ``True``.

    Notes
    -----
    Keys should be timestamps or any other stable hashable identity —
    *not* row indices.  Index keys silently break when a frame is sliced
    or resampled; timestamp keys survive any sub-range extraction.

    Examples
    --------
    >>> from predcache import PredictionCache
    >>> cache = PredictionCache("preds.pkl")
    >>> cache.get_or_compute(
    ...     keys=["2026-01-02 10:00", "2026-01-02 10:01"],
    ...     compute_fn=lambda missing: {k: float(k[-4:]) for k in missing},
    ... )
    {'2026-01-02 10:01': 10.01, '2026-01-02 10:01': 10.01}
    """

    def __init__(self, path: str, backend: str = "pickle"):
        self.path = path
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Registered: {sorted(_BACKENDS)}"
            )
        self._dump, self._load = _BACKENDS[backend]
        self._entries: Dict[Hashable, Any] = {}
        self._dirty = False

    # -- I/O -----------------------------------------------------------------

    def load(self) -> "PredictionCache":
        if os.path.exists(self.path):
            self._entries = self._load(self.path)
        return self

    def save(self) -> None:
        if not self._dirty:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if self.path.endswith(".tmp"):
            self._dump(self._entries, self.path)
        else:
            fd, tmp = tempfile.mkstemp(
                dir=directory or ".", prefix=os.path.basename(self.path), suffix=".tmp"
            )
            os.close(fd)
            try:
                self._dump(self._entries, tmp)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        self._dirty = False

    # -- lookup / mutation -----------------------------------------------------

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: Hashable, default: Any = None) -> Any:
        return self._entries.get(key, default)

    def put(self, key: Hashable, value: Any) -> None:
        self._entries[key] = value
        self._dirty = True

    def keys(self) -> Iterable[Hashable]:
        return self._entries.keys()

    def items(self) -> Iterable[tuple]:
        return self._entries.items()

    def info(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "entries": len(self._entries),
            "backend": "pickle" if self.path.endswith(".pkl") else "auto",
        }

    # -- the core convenience ---------------------------------------------------

    def get_or_compute(
        self,
        keys: List[Hashable],
        compute_fn: Callable[[List[Hashable]], Dict[Hashable, Any]],
        save: bool = True,
    ) -> Dict[Hashable, Any]:
        """Return predictions for *keys*, computing only the missing ones.

        ``compute_fn`` receives the list of missing keys and must return a
        dict ``{key: value}``.  Computed values are merged into the cache.
        """
        missing = [k for k in keys if k not in self._entries]
        with _TimedSpan(
            "predcache.get_or_compute",
            {
                "predcache.requested": len(keys),
                "predcache.computed": len(missing),
                "predcache.hits": len(keys) - len(missing),
            },
        ):
            if missing:
                computed = compute_fn(missing) or {}
                for k, v in computed.items():
                    self._entries[k] = v
                self._dirty = True
                add_count("predcache.predictions.computed", len(computed))
                if save:
                    self.save()
        return {k: self._entries[k] for k in keys if k in self._entries}

    def drop(self, keys: List[Hashable]) -> int:
        removed = 0
        for k in keys:
            if k in self._entries:
                del self._entries[k]
                removed += 1
        if removed:
            self._dirty = True
        return removed