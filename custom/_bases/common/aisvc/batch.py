"""Batching for custom servers: an image supplies one function that runs one padded batch.

    embed = MicroBatcher(model.embed_batch, length=len)   # embed_batch(list[clip]) -> list[vector]
    vectors = embed(clips)                                # any number of clips, results in input order

`MicroBatcher` sorts items by length, runs chunks within `Limits` (max items, and max padded length =
items x longest item, in the image's unit: samples at the model rate for audio, or tokens), and on an
out of memory error empties the CUDA cache, halves the chunk, retries and keeps the lower limit.
`DynamicBatcher` gathers single calls from concurrent request threads (one file per HTTP request) into
one micro-batched call. Limits come from AISVC_MAX_BATCH and AISVC_MAX_BATCH_LEN; GET /platform/memory
reports them.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

log = logging.getLogger("aisvc")

DEFAULT_MAX_BATCH = 8
DEFAULT_WAIT_MS = 10.0
# CTranslate2 ("CUDA failed with error out of memory"), onnxruntime ("Failed to allocate memory",
# BFCArena), cuBLAS/cuDNN allocation failures: these come as RuntimeError, so match the message.
_OOM_MESSAGE = re.compile(r"out of memory|failed to allocate memory|cudaErrorMemoryAllocation|ALLOC_FAILED|BFCArena",
                          re.IGNORECASE)
_LIMITS: dict[str, Limits] = {}  # name -> limits of every batcher in this process, for /platform/memory


def _env_int(var: str, default: int | None) -> int | None:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{var}={raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"{var} must be at least 1, got {value}")
    return value


def is_oom(exc: BaseException) -> bool:
    """True for torch.cuda.OutOfMemoryError, MemoryError, and ONNX Runtime / CTranslate2 out of memory errors."""
    torch = sys.modules.get("torch")  # only an image that already imported torch can raise its error
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    return isinstance(exc, MemoryError) or bool(_OOM_MESSAGE.search(str(exc)))


def free_cuda_cache() -> None:
    """torch.cuda.empty_cache() if torch is loaded and CUDA is up; nothing otherwise."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # never let cleanup hide the real error
        log.debug("empty_cache failed", exc_info=True)


class Limits:
    """Per-batch caps: `max_items` and `max_len` (padded length, items x longest; None = no cap).

    A single item always runs, even when it is longer than max_len. Out of memory backoff lowers
    max_len for the rest of the process; `stats()` shows the configured and current values.
    """

    def __init__(self, max_items: int = DEFAULT_MAX_BATCH, max_len: int | None = None):
        if max_items < 1 or (max_len is not None and max_len < 1):
            raise ValueError(f"batch limits must be at least 1 (max_items={max_items}, max_len={max_len})")
        self.max_items, self.max_len = max_items, max_len
        self.configured = (max_items, max_len)
        self.backoffs = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, max_items: int = DEFAULT_MAX_BATCH, max_len: int | None = None) -> Limits:
        """AISVC_MAX_BATCH and AISVC_MAX_BATCH_LEN, falling back to the image's own defaults."""
        return cls(_env_int("AISVC_MAX_BATCH", max_items), _env_int("AISVC_MAX_BATCH_LEN", max_len))

    def plan(self, lengths: Sequence[int]) -> list[list[int]]:
        """Chunks of positions into `lengths` (which must be sorted ascending), each within the limits."""
        chunks: list[list[int]] = []
        cur: list[int] = []
        max_items, max_len = self.max_items, self.max_len
        for i, n in enumerate(lengths):
            k = len(cur) + 1
            if cur and (k > max_items or (max_len is not None and k * n > max_len)):
                chunks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        return chunks

    def lower(self, padded: int, items: int) -> None:
        """Out of memory on a chunk of `items` items and `padded` padded length: halve max_len."""
        with self._lock:
            new = max(1, padded // 2)
            if self.max_len is None or new < self.max_len:
                self.max_len = new
            self.backoffs += 1
            log.warning("out of memory on a batch of %d items (padded length %d); retrying in smaller chunks, "
                        "max padded length is now %d", items, padded, self.max_len)

    def stats(self) -> dict[str, Any]:
        return {"max_items": self.max_items, "max_len": self.max_len, "configured_max_items": self.configured[0],
                "configured_max_len": self.configured[1], "backoffs": self.backoffs, "lowered": self.backoffs > 0}


def limits_stats() -> dict[str, dict[str, Any]]:
    """Current limits of every batcher in this process, by name."""
    return {name: lim.stats() for name, lim in _LIMITS.items()}


class MicroBatcher:
    """Run any number of items through `fn` (one padded batch -> one result per item) in limited chunks.

    Items are sorted by `length` so each chunk pads little; results come back in input order. Calls
    to `fn` are serialised, so `fn` needs no lock of its own. A chunk that runs out of memory is split
    and retried (see `Limits.lower`); a single item that runs out of memory raises.
    """

    def __init__(self, fn: Callable[[list[Any]], Sequence[Any]], *, length: Callable[[Any], int] = len,
                 limits: Limits | None = None, name: str = "default"):
        self.fn, self.length, self.name = fn, length, name
        self.limits = limits or Limits.from_env()
        self._lock = threading.Lock()
        _LIMITS[name] = self.limits

    def __call__(self, items: Sequence[Any]) -> list[Any]:
        items = list(items)
        lengths = [int(self.length(x)) for x in items]
        order = sorted(range(len(items)), key=lengths.__getitem__)
        out: list[Any] = [None] * len(items)
        todo = deque([order[j] for j in c] for c in self.limits.plan([lengths[i] for i in order]))
        while todo:
            chunk = todo.popleft()
            try:
                with self._lock:
                    results = list(self.fn([items[i] for i in chunk]))
            except Exception as exc:
                if not is_oom(exc):
                    raise
                if len(chunk) == 1:
                    free_cuda_cache()
                    raise
            else:
                if len(results) != len(chunk):
                    raise RuntimeError(f"batch function returned {len(results)} results for {len(chunk)} items")
                for i, r in zip(chunk, results):
                    out[i] = r
                continue
            # Out of memory on several items. Outside the except block, so the traceback's tensors are freed.
            # max_len drops below this chunk's padded length, so planning it again always splits it.
            free_cuda_cache()
            self.limits.lower(len(chunk) * max(lengths[i] for i in chunk), len(chunk))
            parts = self.limits.plan([lengths[i] for i in chunk])
            todo.extendleft(reversed([[chunk[j] for j in p] for p in parts]))
        return out


class _Call:
    __slots__ = ("item", "key", "at", "done", "result", "error")

    def __init__(self, item: Any, key: Hashable):
        self.item, self.key, self.at = item, key, time.monotonic()
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class DynamicBatcher:
    """Gather single calls that arrive within `wait_ms` of each other into one micro-batched call.

    For routes that take one input per HTTP request. Call it from a sync route (the aisvc factories'
    routes are sync, so FastAPI runs each request in a worker thread); it blocks until that caller's
    result is ready. Only calls with equal `key(item)` share a batch (e.g. same language and options).
    An error in a batch is raised to every caller in it, after out of memory backoff. Validate and
    decode each request before calling, so one bad file fails only its own request.
    """

    def __init__(self, batch: MicroBatcher | Callable[[list[Any]], Sequence[Any]], *,
                 key: Callable[[Any], Hashable] = lambda item: None, wait_ms: float | None = None, **micro: Any):
        self.batch = batch if isinstance(batch, MicroBatcher) else MicroBatcher(batch, **micro)
        self.key = key
        if wait_ms is None:
            raw = os.environ.get("AISVC_BATCH_WAIT_MS", "").strip()
            wait_ms = float(raw) if raw else DEFAULT_WAIT_MS
        self.wait = max(0.0, wait_ms) / 1000
        self._pending: list[_Call] = []
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None

    def __call__(self, item: Any) -> Any:
        call = _Call(item, self.key(item))
        with self._cond:
            self._pending.append(call)
            if self._worker is None:
                self._worker = threading.Thread(target=self._loop, name=f"batch-{self.batch.name}", daemon=True)
                self._worker.start()
            self._cond.notify()
        call.done.wait()
        if call.error is not None:
            raise call.error
        return call.result

    def _take(self) -> list[_Call]:
        """Wait for a first call, then up to `wait` after it, and take every pending call with its key."""
        with self._cond:
            while not self._pending:
                self._cond.wait()
            first = self._pending[0]
            while (left := first.at + self.wait - time.monotonic()) > 0:
                self._cond.wait(left)
            group = [c for c in self._pending if c.key == first.key]
            self._pending = [c for c in self._pending if c.key != first.key]
            return group

    def _loop(self) -> None:
        while True:
            group = self._take()
            try:
                results = self.batch([c.item for c in group])
            except BaseException as exc:
                for c in group:
                    c.error = exc
            else:
                for c, r in zip(group, results):
                    c.result = r
            for c in group:
                c.done.set()


def pad(rows: Sequence[np.ndarray], value: float = 0.0, dtype: Any = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Stack 1-D rows into a [N, longest] array padded with `value`, plus their lengths (int64 [N]).

    An attention mask is `np.arange(padded.shape[1]) < lengths[:, None]`.
    """
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    out = np.full((len(rows), int(lengths.max(initial=0))), value, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out, lengths


def unpad(padded: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Rows of an OIP padded batch (AUDIO [N, T] with LENGTHS [N]) cut to their lengths."""
    return [padded[i, : int(n)] for i, n in enumerate(np.asarray(lengths).reshape(-1))]
