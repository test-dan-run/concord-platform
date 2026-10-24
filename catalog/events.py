"""Publish/subscribe for the SSE stream: threads publish, each subscriber drains its own queue."""

from __future__ import annotations

import queue
import threading
from typing import Any

MAX_BACKLOG = 1000


class Events:
    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(MAX_BACKLOG)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((event, data))
            except queue.Full:  # a stalled client: drop it rather than grow without bound
                self.unsubscribe(q)
