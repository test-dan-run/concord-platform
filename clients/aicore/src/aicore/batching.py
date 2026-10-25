"""Sending work to a service in batches of the size the catalog recommends.

    from aicore.batching import batch_size, map_batches, map_concurrent

    n = batch_size("wavlm-sv", item_s=8.0)                 # the catalog's recommendation, else `default`
    vectors = map_batches(embed_padded, clips, n)          # embed_padded(list of clips) -> list, one call per batch
    texts = map_concurrent(transcribe_one, files, n)       # n requests at once: dynamic batching and vLLM batch them

A service whose API takes many items per request (OIP `AUDIO [-1,-1]` + `LENGTHS`) gets `map_batches`; a service
that takes one item per request (OpenAI transcriptions, chat) gets `map_concurrent`, and the server or vLLM joins
requests that arrive together. Results come back in input order. The recommendation is for the memory the
service gets at its start; without the catalog, or before the build has a batch profile, `default` is used, and
the server's out of memory backoff keeps a batch that is too large from failing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")
log = logging.getLogger("aicore.batching")
DEFAULT = 8
MAX = 256


@lru_cache(maxsize=256)
def _recommended(service: str, item_s: float | None, item_tokens: int | None, catalog_url: str | None) -> int | None:
    from .catalog import Catalog, CatalogError

    try:
        cat = Catalog(catalog_url, timeout=5.0)
        rec = cat.batch(service, item_s=item_s, item_tokens=item_tokens).get("recommendation")
    except (CatalogError, OSError, ValueError) as exc:
        log.debug("no batch recommendation for %s: %s", service, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - a batch size is never worth failing a job over
        log.debug("no batch recommendation for %s: %s", service, exc)
        return None
    return int(rec["batch"]) if rec and rec.get("batch") else None


def batch_size(service: str, *, item_s: float | None = None, item_tokens: int | None = None, default: int = DEFAULT,
               catalog: Any | None = None) -> int:
    """The catalog's recommended batch for `service` and items of `item_s` seconds (audio) or `item_tokens` tokens
    (vLLM), at the memory the service gets; `default` when there is none. Cached per process; `catalog` (a
    Catalog or FakeCatalog) skips the cache, for tests."""
    if item_s is None and item_tokens is None:
        item_s = 20.0
    if catalog is not None:
        rec = (catalog.batch(service, item_s=item_s, item_tokens=item_tokens) or {}).get("recommendation")
        n = int(rec["batch"]) if rec and rec.get("batch") else None
    else:
        from .endpoints import catalog_url

        url = catalog_url()
        n = _recommended(service, item_s, item_tokens, url) if url else None
    return max(1, min(MAX, n or default))


def batches(items: Sequence[T], size: int) -> Iterable[list[T]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def map_batches(fn: Callable[[list[T]], Sequence[R]], items: Sequence[T], size: int) -> list[R]:
    """fn(batch) -> one result per item, called once per batch of at most `size` items, in order."""
    out: list[R] = []
    for chunk in batches(items, size):
        res = list(fn(chunk))
        if len(res) != len(chunk):
            raise ValueError(f"batch function returned {len(res)} results for {len(chunk)} items")
        out.extend(res)
    return out


def map_concurrent(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """fn(item) for every item, `workers` requests at a time; results in input order, the first error raised."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(workers, len(items))) as ex:
        return list(ex.map(fn, items))
