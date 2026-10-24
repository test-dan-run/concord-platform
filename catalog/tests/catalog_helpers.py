"""Helpers for catalog tests: drive the scheduler by hand."""

from __future__ import annotations

import time


def drain(c) -> None:
    """Run the starter thread's queued stops and starts in this thread."""
    s = c.scheduler
    while not s._jobs.empty():
        kind, name = s._jobs.get_nowait()
        if kind == "start":
            s._do_start(name)
        elif kind == "stop":
            s._do_stop(name)


def step(c, n: int = 2) -> None:
    for _ in range(n):
        c.scheduler.tick()
        drain(c)
    c.scheduler.tick()


def wait_builds(c, names: list[str], timeout: float = 10.0) -> None:
    """Tick until every named model has a build key (weights are hashed on a background thread)."""
    deadline = time.monotonic() + timeout
    while True:
        c.scheduler.tick()
        with c.scheduler.lock:
            if all(c.scheduler.runs[n].build and c.scheduler.runs[n].build.build_key for n in names):
                return
        if time.monotonic() > deadline:
            raise AssertionError({n: c.scheduler.runs[n].build for n in names})
        time.sleep(0.02)
