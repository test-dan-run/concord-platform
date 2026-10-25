"""A subscriber that falls too far behind is dropped, and its event stream then ends, so the client
reconnects and gets a fresh snapshot instead of keep-alives and no events."""

import asyncio

from catalog.app import create_app
from catalog.events import MAX_BACKLOG


class OpenRequest:
    """A client that never disconnects."""

    async def is_disconnected(self) -> bool:
        return False


def test_the_stream_of_a_dropped_subscriber_ends_after_its_backlog(cat):
    app = create_app(cat, start=False)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/events")
    snapshot = len(cat.registry.records(include_all=True))

    async def read_all() -> list[str]:
        response = await endpoint(OpenRequest())  # subscribes now; the stream has not started
        for i in range(MAX_BACKLOG + 1):  # one more than the backlog holds: the subscriber is dropped
            cat.events.publish("lease", {"i": i})
        cat.events.publish("lease", {"i": "after the drop"})
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(asyncio.wait_for(read_all(), timeout=5))
    assert chunks[0] == "retry: 3000\n\n"
    assert sum(c.startswith("event: model\n") for c in chunks) == snapshot
    leases = [c for c in chunks if c.startswith("event: lease\n")]
    assert len(leases) == MAX_BACKLOG and '"i": 0}' in leases[0] and f'"i": {MAX_BACKLOG - 1}}}' in leases[-1]


def test_a_subscriber_that_keeps_up_is_not_dropped(cat):
    q = cat.events.subscribe()
    try:
        for i in range(3 * MAX_BACKLOG):
            cat.events.publish("lease", {"i": i})
            q.get_nowait()
        cat.events.publish("lease", {"i": "last"})
        assert q.get_nowait() == ("lease", {"i": "last"})
    finally:
        cat.events.unsubscribe(q)
