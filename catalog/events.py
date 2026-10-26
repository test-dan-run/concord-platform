"""Publish/subscribe for the SSE stream: threads publish, each subscriber drains its own queue."""

from __future__ import annotations

import queue
import threading
from typing import Any

MAX_BACKLOG = 1000


class Subscription(queue.Queue):
    """One subscriber's queue. `dropped` is set when its backlog was full: it gets no more events."""

    def __init__(self) -> None:
        super().__init__(MAX_BACKLOG)
        self.dropped = False


class Events:
    def __init__(self) -> None:
        self._subs: set[Subscription] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> Subscription:
        q = Subscription()
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
                q.dropped = True
                self.unsubscribe(q)
