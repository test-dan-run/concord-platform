"""A manual start's pin waits on the request that brings the model up: its own start request, or the live request
it joined. If that request ends before the model runs, the pin goes, even when another request names the model."""

from catalog_helpers import step

M = "whisper-large-v3"


def test_a_pin_from_a_start_that_never_ran_is_dropped_when_a_lease_for_it_is_queued(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    cat.settings.gateway_idle_s = 0.0
    blocker = s.request("label:job-3", ["qwen3-32b-awq"])
    step(cat)
    start = s.request("manual", [M], kind="start")  # `just up whisper-large-v3` while blocked
    lease = s.request("bench:job-1", [M])  # queued behind the blocker too
    step(cat)
    assert s.get(start.id).state == "queued" and s.runs[M].pinned
    s.requests[start.id].heartbeat_at -= 3600  # `just up` gave up
    s.tick()
    assert s.get(start.id).state == "expired" and not s.runs[M].pinned
    s.release(blocker.id)
    step(cat)
    assert s.get(lease.id).state == "active"
    s.release(lease.id)
    step(cat)
    step(cat)
    assert not s.runs[M].running and M not in runtime.running and runtime.stopped[-1] == M


def test_a_pin_that_joined_a_queued_lease_is_dropped_when_that_lease_ends_before_the_model_runs(cat, runtime):
    s = cat.scheduler
    blocker = s.request("label:job-3", ["qwen3-32b-awq"])
    step(cat)
    first = s.request("bench:job-1", [M])
    assert s.ensure_started(M, holder="manual").id == first.id and s.runs[M].pinned  # `just up` joins it
    second = s.request("bench:job-2", [M])
    s.release(first.id)  # before the model ran; the second lease still names it
    s.tick()
    assert s.get(second.id).state == "queued" and not s.runs[M].pinned
    s.release(blocker.id)
    step(cat)
    assert s.get(second.id).state == "active" and not s.runs[M].pinned


def test_a_pin_that_joined_a_queued_lease_stays_while_that_lease_lives(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    cat.settings.gateway_idle_s = 0.0
    blocker = s.request("label:job-3", ["qwen3-32b-awq"])
    step(cat)
    lease = s.request("bench:job-1", [M])
    assert s.ensure_started(M, holder="manual").id == lease.id
    step(cat)
    assert s.get(lease.id).state == "queued" and s.runs[M].pinned
    s.release(blocker.id)
    step(cat)
    assert s.get(lease.id).state == "active" and s.runs[M].pinned
    s.release(lease.id)
    step(cat)
    assert s.is_up(M) and s.runs[M].pinned
