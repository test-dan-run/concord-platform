"""Scheduler rules: leases, strict first-come-first-served queue, fit, cost-aware eviction."""

import time

from catalog_helpers import drain, step

from catalog.db import now_iso
from catalog.scheduler import Scheduler


def cold(c, model: str, seconds: float) -> None:
    """Make every measured load of the model (the fake runtime's are ~0 s) take `seconds`."""
    c.db.execute("UPDATE cold_start SET seconds = ? WHERE model_id = ?", (seconds, model))
    c.db.execute("INSERT INTO cold_start (model_id, build_key, seconds, at) VALUES (?, NULL, ?, ?)",
                 (model, seconds, now_iso()))
    c.scheduler._cold_cache.clear()


def run_unleased(c, *models: str) -> None:
    """Models running with nobody holding them, as after a gateway start or `just up`."""
    for m in models:
        r = c.scheduler.request("manual", [m], kind="start")
        step(c)
        assert c.scheduler.get(r.id).state == "done"
        c.scheduler.runs[m].ready_at = time.time() - 3600  # up for an hour: settled


def test_lease_starts_its_models_and_becomes_active(cat, runtime):
    s = cat.scheduler
    r = s.request("bench:job-1", ["default-asr", "silero-vad"])
    assert r.models == ["whisper-large-v3", "silero-vad"] and r.state == "starting"
    step(cat)
    assert s.get(r.id).state == "active" and sorted(runtime.started) == ["silero-vad", "whisper-large-v3"]
    assert s.model_view("whisper-large-v3")["leased_by"] == ["bench:job-1"]
    assert cat.db.one("SELECT COUNT(*) AS n FROM cold_start WHERE model_id = 'whisper-large-v3'")["n"] == 1


def test_a_leased_model_is_never_stopped_and_the_waiter_explains_why(cat, runtime):
    s = cat.scheduler
    a = s.request("label:job-3", ["qwen3-32b-awq"])
    step(cat)
    b = s.request("bench:job-12", ["whisper-large-v3"])
    step(cat)
    b = s.get(b.id)
    assert b.state == "queued" and b.position == 1
    assert "qwen3-32b-awq (leased by label:job-3)" in b.waiting_for
    assert s.model_view("whisper-large-v3")["state"] == "queued" and "qwen3-32b-awq" in runtime.running
    s.release(a.id)
    step(cat)
    assert s.get(b.id).state == "active" and "qwen3-32b-awq" in runtime.stopped


def test_eviction_stops_the_cheapest_reload_and_only_enough(cat, runtime):
    run_unleased(cat, "qwen3-8b", "qwen2-5-14b-instruct-awq")
    cold(cat, "qwen3-8b", 100.0)
    cold(cat, "qwen2-5-14b-instruct-awq", 40.0)
 # 0.35 + 0.45 of the GPU
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])  # + 0.15 > util_limit 0.93
    step(cat)
    assert cat.scheduler.get(r.id).state == "active"
    assert runtime.stopped == ["qwen2-5-14b-instruct-awq"] and "qwen3-8b" in runtime.running


def test_calls_in_flight_protect_a_model(cat, runtime):
    run_unleased(cat, "qwen3-8b", "qwen2-5-14b-instruct-awq")
    cold(cat, "qwen3-8b", 100.0)
    cold(cat, "qwen2-5-14b-instruct-awq", 40.0)
    cat.scheduler.begin_call("qwen2-5-14b-instruct-awq")  # the cheap one is busy
    step(cat)
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    assert cat.scheduler.get(r.id).state == "active" and runtime.stopped == ["qwen3-8b"]


def test_settle_time_keeps_a_fresh_model_running(cat, runtime):
    run_unleased(cat, "qwen2-5-14b-instruct-awq", "qwen3-8b")
    cat.settings.settle_min_s = 7200.0
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "queued" and "settling" in r.waiting_for and runtime.stopped == []


def test_queue_is_strictly_first_come_first_served(cat, runtime):
    s = cat.scheduler
    s.request("label:job-3", ["qwen2-5-14b-instruct-awq", "qwen3-8b"])  # 0.80, leased
    step(cat)
    head = s.request("bench:job-1", ["qwen3-32b-awq"])  # cannot fit while that lease lives
    later = s.request("bench:job-2", ["bge-m3"])  # 0.08 would fit, but must wait its turn
    cpu = s.request("bench:job-3", ["silero-vad"])  # takes no GPU memory: may pass
    step(cat)
    assert s.get(head.id).state == "queued" and s.get(head.id).position == 1
    later = s.get(later.id)
    assert later.state == "queued" and later.position == 2 and later.waiting_for.startswith("queued behind bench:job-1")
    assert s.get(cpu.id).state == "active"


def test_a_holder_cannot_wait_on_its_own_lease(cat):
    s = cat.scheduler
    s.request("label:job-3", ["qwen2-5-14b-instruct-awq", "qwen3-8b"])
    step(cat)
    r = s.request("label:job-3", ["qwen3-32b-awq"])
    step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "release that lease" in r.error


def test_a_set_that_can_never_fit_fails_at_once(cat):
    r = cat.scheduler.request("bench:job-1", ["qwen3-32b-awq", "whisper-large-v3"])
    assert cat.scheduler.get(r.id).state == "failed" and "never run at once" in cat.scheduler.get(r.id).error


def test_missed_heartbeats_expire_a_lease_and_free_its_models(cat):
    s = cat.scheduler
    r = s.request("bench:job-1", ["whisper-large-v3"], ttl_s=0.05)
    step(cat)
    time.sleep(0.1)
    s.tick()
    assert s.get(r.id).state == "expired" and s.model_view("whisper-large-v3")["leased_by"] == []


def test_a_failed_start_fails_the_lease(cat, runtime):
    runtime.fail.add("whisper-large-v3")
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "failed" and "failed to start" in r.error
    assert cat.scheduler.model_view("whisper-large-v3")["state"] == "failed"


def test_a_leased_model_that_dies_is_started_again(cat, runtime):
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    runtime.running.pop("whisper-large-v3")  # crashed, or stopped by hand
    step(cat)
    assert cat.scheduler.get(r.id).state == "active" and runtime.started.count("whisper-large-v3") == 2


def test_live_leases_survive_a_catalog_restart(cat, settings, runtime):
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    again = Scheduler(cat.db, settings, runtime, cat.builds, cat.events)
    again.platform()
    assert again.get(r.id).state == "active" and again.get(r.id).holder == "bench:job-1"


def test_gateway_start_requests_end_once_the_model_is_up(cat, runtime):
    s = cat.scheduler
    r = s.ensure_started("whisper-large-v3")
    assert r is not None and r.kind == "start" and s.ensure_started("whisper-large-v3") is r  # no duplicate
    drain(cat)
    step(cat)
    assert s.get(r.id).state == "done" and s.is_up("whisper-large-v3") and s.ensure_started("whisper-large-v3") is None
    assert s.model_view("whisper-large-v3")["leased_by"] == []  # a start protects nothing once it is done


def test_a_lease_is_active_only_once_its_models_have_build_keys(cat, assets):
    r = cat.scheduler.request("bench:job-1", ["qwen3-5-4b"])  # its weights folder is not in this AI_ASSETS
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "starting" and "weights folder" in r.waiting_for and cat.scheduler.is_up("qwen3-5-4b")
    (assets / "qwen3.5-4b").mkdir()
    (assets / "qwen3.5-4b" / "model.safetensors").write_bytes(b"w")
    cat.builds.hasher._quick.clear()
    deadline = time.time() + 10
    while cat.scheduler.get(r.id).state != "active":
        assert time.time() < deadline
        step(cat, 1)
        time.sleep(0.02)


# Spin-down: a model a lease brought up is stopped once no request needs it.

def test_a_model_a_lease_brought_up_is_stopped_once_nothing_needs_it(cat, runtime):
    cat.settings.release_linger_s = 0.0
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3", "silero-vad"])
    step(cat)
    cat.scheduler.release(r.id)
    step(cat)
    assert sorted(runtime.stopped) == ["silero-vad", "whisper-large-v3"] and runtime.running == {}
    assert cat.scheduler.model_view("whisper-large-v3")["state"] == "stopped"


def test_the_linger_keeps_a_model_for_a_back_to_back_lease(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 3600.0
    s.release(s.request("bench:job-1", ["whisper-large-v3"]).id)
    step(cat)
    assert runtime.stopped == []
    r = s.request("bench:job-2", ["whisper-large-v3"])
    step(cat)
    assert s.get(r.id).state == "active" and runtime.started == ["whisper-large-v3"]  # not loaded again
    cat.settings.release_linger_s = 0.0
    s.release(r.id)
    step(cat)
    assert runtime.stopped == ["whisper-large-v3"]


def test_a_model_another_request_needs_stays_up(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    a = s.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    b = s.request("label:job-2", ["whisper-large-v3"])
    step(cat)
    s.release(a.id)
    step(cat)
    assert s.get(b.id).state == "active" and runtime.stopped == []
    s.release(b.id)
    step(cat)
    assert runtime.stopped == ["whisper-large-v3"]


def test_a_model_started_on_purpose_stays_up_after_a_lease(cat, runtime):
    cat.settings.release_linger_s = 0.0
    run_unleased(cat, "whisper-large-v3")  # a manual start or `just up`
    cat.scheduler.release(cat.scheduler.request("bench:job-1", ["whisper-large-v3"]).id)
    step(cat)
    assert runtime.stopped == [] and cat.scheduler.is_up("whisper-large-v3")


def test_calls_in_flight_hold_off_the_spin_down(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    r = s.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    s.begin_call("whisper-large-v3")
    s.release(r.id)
    step(cat)
    assert runtime.stopped == []
    s.end_call("whisper-large-v3", error=False)
    step(cat)
    assert runtime.stopped == ["whisper-large-v3"]


def test_a_released_model_is_stopped_for_room_before_its_settle_time(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 3600.0
    cat.settings.settle_min_s = 7200.0
    s.release(s.request("bench:job-1", ["qwen2-5-14b-instruct-awq"]).id)  # 0.45, just loaded
    step(cat)
    r = s.request("bench:job-2", ["qwen3-32b-awq"])  # 0.90: needs that room
    step(cat)
    assert s.get(r.id).state == "active" and runtime.stopped == ["qwen2-5-14b-instruct-awq"]


# Measured GPU memory: a model starts only when its whole budget is free on the device.

def test_measured_memory_stops_more_than_declared_footprints_would(cat, runtime):
    runtime.gpu_total_gib = 31.84
    runtime.gpu_other_gib = 8.0  # another process on the GPU
    run_unleased(cat, "qwen3-8b")  # 0.35 = 11.1 GiB
    # 0.35 + 0.45 fits util_limit, but 14.3 GiB + margin is not free next to 8 GiB of other use
    r = cat.scheduler.request("bench:job-1", ["qwen2-5-14b-instruct-awq"])
    step(cat)
    assert cat.scheduler.get(r.id).state == "active" and runtime.stopped == ["qwen3-8b"]


def test_a_model_waits_rather_than_loading_partly(cat, runtime):
    s = cat.scheduler
    runtime.gpu_total_gib = 31.84
    runtime.gpu_other_gib = 20.0
    r = s.request("bench:job-1", ["qwen2-5-14b-instruct-awq"])
    step(cat)
    r = s.get(r.id)
    assert r.state == "queued" and "to load entirely" in r.waiting_for and "outside the platform" in r.waiting_for
    assert runtime.started == []
    runtime.gpu_other_gib = 1.5
    step(cat)
    assert s.get(r.id).state == "active"


def test_no_gpu_model_starts_while_another_is_loading(cat, runtime):
    s = cat.scheduler
    runtime.gpu_total_gib = 31.84
    s.tick()
    s.runs["bge-m3"].starting = True
    r = s.request("bench:job-1", ["whisper-large-v3"])
    assert r.state == "queued" and "bge-m3 to finish loading" in r.waiting_for
    s.runs["bge-m3"].starting = False
    step(cat)
    assert s.get(r.id).state == "active"


def test_a_start_that_finds_too_little_free_memory_queues_again(cat, runtime):
    s = cat.scheduler
    cat.settings.gpu_free_wait_s = 0.0
    runtime.gpu_total_gib = 31.84
    s.tick()  # measured: all free
    runtime.gpu_other_gib = 28.0  # taken before the start runs
    r = s.request("bench:job-1", ["whisper-large-v3"])
    assert r.state == "starting"
    drain(cat)
    s.tick()
    r = s.get(r.id)
    assert r.state == "queued" and "to load entirely" in r.waiting_for and runtime.started == []
    runtime.gpu_other_gib = 0.0
    step(cat)
    assert s.get(r.id).state == "active"


def test_a_model_the_gateway_started_stops_after_the_idle_timeout(cat, runtime):
    s = cat.scheduler
    cat.settings.gateway_idle_s = 3600.0
    s.ensure_started("whisper-large-v3")  # a call without a lease
    step(cat)
    s.begin_call("whisper-large-v3")
    s.end_call("whisper-large-v3", error=False)
    step(cat)
    assert s.is_up("whisper-large-v3") and runtime.stopped == []
    cat.settings.gateway_idle_s = 0.0
    step(cat)
    assert runtime.stopped == ["whisper-large-v3"]


def test_a_manual_start_has_no_idle_timeout(cat, runtime):
    cat.settings.gateway_idle_s = 0.0
    cat.scheduler.ensure_started("whisper-large-v3", holder="manual")
    step(cat)
    assert cat.scheduler.is_up("whisper-large-v3") and runtime.stopped == []
