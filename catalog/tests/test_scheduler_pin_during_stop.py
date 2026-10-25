"""A manual start (the UI's Start, `just up`) that arrives while its model is being stopped keeps its pin: the stop
ends the old run, not the new wish to keep the model up."""

from catalog_helpers import drain, step

M = "whisper-large-v3"


def spun_down(cat):
    """A model whose lease ended and whose stop is queued but not yet run by the starter."""
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    r = s.request("bench:job-1", [M])
    step(cat)
    s.release(r.id)
    s.tick()
    run = s.runs[M]
    assert run.stopping and not run.pinned
    return run


def test_a_manual_start_queued_behind_a_stop_keeps_its_pin(cat, runtime):
    s = cat.scheduler
    run = spun_down(cat)
    start = s.ensure_started(M, holder="manual")
    assert start is not None and start.kind == "start" and run.pinned
    drain(cat)  # the stop, then the start
    step(cat)
    assert s.get(start.id).state == "done" and s.is_up(M) and run.pinned
    cat.settings.gateway_idle_s = 0.0  # nothing protects an unpinned model now
    step(cat)
    assert s.is_up(M) and runtime.stopped == [M] and runtime.started == [M, M]


def test_a_manual_start_for_a_model_a_lease_brings_back_keeps_its_pin(cat, runtime):
    s = cat.scheduler
    run = spun_down(cat)
    lease = s.request("bench:job-2", [M])  # starts it again after the stop
    assert s.ensure_started(M, holder="manual").id == lease.id
    drain(cat)
    step(cat)
    assert s.get(lease.id).state == "active" and run.pinned
    s.release(lease.id)
    step(cat)
    assert s.is_up(M) and runtime.stopped == [M]


def test_a_stop_asked_for_after_the_pin_still_unpins(cat, runtime):
    s = cat.scheduler
    start = s.ensure_started(M, holder="manual")
    step(cat)
    assert s.get(start.id).state == "done" and s.runs[M].pinned
    s.stop_model(M)
    drain(cat)
    step(cat)
    assert not s.is_up(M) and not s.runs[M].pinned and runtime.stopped == [M]
