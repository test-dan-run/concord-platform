"""A stop that keeps failing is not tried on every tick: the spin-down tries again on the next tick once, then waits
5 s, doubling after each failure up to 5 minutes. A stop that works resets the wait."""

import logging
import time
import types

import pytest
from catalog_helpers import drain, step

import catalog.scheduler as scheduler_module

M = "whisper-large-v3"


@pytest.fixture
def clock(monkeypatch):
    """The scheduler's wall clock, moved by hand."""
    c = types.SimpleNamespace(t=time.time())
    monkeypatch.setattr(scheduler_module, "time", types.SimpleNamespace(time=lambda: c.t, monotonic=time.monotonic))
    return c


def failing_stop(runtime):
    calls = []
    real = runtime.stop

    def stop(p, names):
        calls.append(names)
        raise RuntimeError("docker compose stop whisper-large-v3 exited 1: error during connect")

    runtime.stop = stop
    return calls, real


def test_a_failing_stop_waits_longer_after_each_failure(cat, runtime, clock, caplog):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    cat.settings.gateway_idle_s = 10 ** 6
    calls, real = failing_stop(runtime)
    r = s.request("bench:job-1", [M])
    step(cat)
    s.release(r.id)
    run = s.runs[M]
    with caplog.at_level(logging.WARNING, logger="catalog.scheduler"):
        s.tick()
        drain(cat)  # the first failure: tried again on the next tick
        s.tick()
        assert run.stopping
        drain(cat)
        for wait in [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0, 300.0]:
            assert not run.stopping and run.running
            tried = len(calls)
            for _ in range(3):  # ticks inside the wait try nothing
                s.tick()
                drain(cat)
            clock.t += wait - 0.5
            s.tick()
            assert not run.stopping and len(calls) == tried, wait
            clock.t += 1.0
            s.tick()
            assert run.stopping, wait
            drain(cat)
    failed = [rec for rec in caplog.records if "stop of whisper-large-v3 failed" in rec.getMessage()]
    assert len(failed) == len(calls) == 11  # one warning per try, not one per tick
    runtime.stop = real
    clock.t += 300.0
    step(cat)
    assert not run.running and runtime.stopped == [M]

    # A stop that worked resets the wait: the next failure is tried again on the next tick.
    calls, real = failing_stop(runtime)
    r = s.request("bench:job-2", [M])
    step(cat)
    s.release(r.id)
    s.tick()
    drain(cat)
    s.tick()
    assert run.stopping and len(calls) == 1


def test_a_manual_stop_does_not_wait_for_the_retry(cat, runtime, clock):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    calls, real = failing_stop(runtime)
    r = s.request("bench:job-1", [M])
    step(cat)
    s.release(r.id)
    for _ in range(2):
        s.tick()
        drain(cat)
    s.tick()
    assert not s.runs[M].stopping and len(calls) == 2  # waiting before the third try
    runtime.stop = real
    s.stop_model(M)
    drain(cat)
    step(cat)
    assert not s.is_up(M) and runtime.stopped == [M]


def test_a_model_that_went_down_outside_the_catalog_starts_with_no_wait(cat, runtime, clock):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    calls, real = failing_stop(runtime)
    r = s.request("bench:job-1", [M])
    step(cat)
    s.release(r.id)
    for _ in range(2):
        s.tick()
        drain(cat)
    assert s.runs[M].stop_failures == 2
    runtime.running.pop(M)  # `docker stop` by hand
    s.tick()
    assert not s.is_up(M) and s.runs[M].stop_failures == 0 and s.runs[M].stop_retry_at is None
