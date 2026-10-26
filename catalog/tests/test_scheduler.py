"""Scheduler rules: leases, strict first-come-first-served queue, one GPU model at a time, memory at start."""

import time

import pytest
from catalog_helpers import drain, step

from catalog.db import now_iso
from catalog.scheduler import Scheduler, SchedulerError
from tools.vram_check import UTIL_VAR, VRAM_VAR


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


def found_running(c, runtime, *models: str) -> None:
    """Models found running outside the catalog (several GPU models at once, as before one-at-a-time)."""
    for m in models:
        runtime.running[m] = 0.0
    c.scheduler.tick()
    for m in models:
        c.scheduler.runs[m].ready_at = time.time() - 3600


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
    assert s.get(b.id).state == "active" and runtime.stopped == ["qwen3-32b-awq"]
    assert runtime.started == ["qwen3-32b-awq", "whisper-large-v3"] and s.swaps == 1


def test_a_gpu_model_stops_the_running_one_and_cpu_models_stay(cat, runtime):
    run_unleased(cat, "qwen3-8b", "silero-vad")
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])  # 0.35 + 0.15 would have fit side by side
    step(cat)
    assert cat.scheduler.get(r.id).state == "active"
    assert runtime.stopped == ["qwen3-8b"] and "silero-vad" in runtime.running and cat.scheduler.swaps == 1


def test_every_other_gpu_model_is_stopped_or_none(cat, runtime):
    found_running(cat, runtime, "qwen3-8b", "qwen2-5-14b-instruct-awq")
    cat.scheduler.begin_call("qwen2-5-14b-instruct-awq")  # busy: so nothing is stopped yet
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "queued" and "qwen2-5-14b-instruct-awq (1 call(s) in flight)" in r.waiting_for
    assert runtime.stopped == []
    cat.scheduler.end_call("qwen2-5-14b-instruct-awq", error=False)
    step(cat)
    assert cat.scheduler.get(r.id).state == "active"
    assert sorted(runtime.stopped) == ["qwen2-5-14b-instruct-awq", "qwen3-8b"] and cat.scheduler.swaps == 1


def test_settle_time_keeps_a_fresh_model_running(cat, runtime):
    run_unleased(cat, "qwen2-5-14b-instruct-awq")  # up for an hour
    cold(cat, "qwen2-5-14b-instruct-awq", 3600.0)  # settle time: 2 x cold start
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "queued" and "settling" in r.waiting_for and runtime.stopped == []


def test_queue_is_strictly_first_come_first_served(cat, runtime):
    s = cat.scheduler
    s.request("label:job-3", ["qwen2-5-14b-instruct-awq"])  # leased
    step(cat)
    head = s.request("bench:job-1", ["qwen3-32b-awq"])  # cannot start while that lease lives
    later = s.request("bench:job-2", ["bge-m3"])  # must wait its turn
    cpu = s.request("bench:job-3", ["silero-vad"])  # takes no GPU memory: may pass
    same = s.request("bench:job-4", ["qwen2-5-14b-instruct-awq"])  # already up and held: may pass
    step(cat)
    assert s.get(head.id).state == "queued" and s.get(head.id).position == 1
    later = s.get(later.id)
    assert later.state == "queued" and later.position == 2 and later.waiting_for.startswith("queued behind bench:job-1")
    assert s.get(cpu.id).state == "active" and s.get(same.id).state == "active"


def test_a_holder_cannot_wait_on_its_own_lease(cat):
    s = cat.scheduler
    s.request("label:job-3", ["qwen3-4b-instruct-2507"])
    step(cat)
    r = s.request("label:job-3", ["whisper-large-v3"])  # 0.25 + 0.15 would have fit side by side
    step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "release that lease" in r.error


def test_two_gpu_models_in_one_request_are_refused(cat):
    with pytest.raises(SchedulerError) as exc:
        cat.scheduler.request("bench:job-1", ["qwen3-4b-instruct-2507", "default-asr"])  # small enough to pack
    assert exc.value.status == 422 and "lease one GPU model per stage" in str(exc.value)
    assert cat.scheduler.live_requests() == []


def test_the_swap_count_survives_a_restart(cat, settings, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 3600.0
    for holder, model in [("a", "qwen3-8b"), ("b", "whisper-large-v3"), ("c", "whisper-large-v3"), ("d", "bge-m3")]:
        r = s.request(holder, [model])
        step(cat)
        assert s.get(r.id).state == "active"
        s.release(r.id)
    assert s.swaps == 2  # 8b -> whisper, whisper -> bge-m3; the second whisper lease found it loaded
    again = Scheduler(cat.db, settings, runtime, cat.builds, cat.events)
    assert again.swaps == 2


def test_missed_heartbeats_expire_a_lease_and_free_its_models(cat):
    s = cat.scheduler
    r = s.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    s.requests[r.id].heartbeat_at -= 3600  # no heartbeat for an hour
    s.tick()
    assert s.get(r.id).state == "expired" and s.model_view("whisper-large-v3")["leased_by"] == []


def test_a_failed_start_fails_the_lease(cat, runtime):
    runtime.fail.add("whisper-large-v3")
    r = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    r = cat.scheduler.get(r.id)
    assert r.state == "failed" and "failed to start" in r.error
    assert cat.scheduler.model_view("whisper-large-v3")["state"] == "failed"


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


# Measured GPU memory: a model starts only when its minimum is free, and gets what is free then.

def test_a_gpu_model_gets_the_memory_measured_free_at_its_start(cat, runtime):
    runtime.gpu_total_gib = 31.84
    runtime.gpu_other_gib = 8.0  # another process on the GPU (Windows keeps some for the display)
    run_unleased(cat, "qwen3-8b")
    assert runtime.envs["qwen3-8b"] == {UTIL_VAR: "0.717", VRAM_VAR: "22.82"}  # (31.84 - 8 - 1) / 31.84
    r = cat.scheduler.request("bench:job-1", ["qwen2-5-14b-instruct-awq", "silero-vad"])
    step(cat)
    assert cat.scheduler.get(r.id).state == "active" and runtime.stopped == ["qwen3-8b"]
    assert runtime.envs["qwen2-5-14b-instruct-awq"][UTIL_VAR] == "0.717" and runtime.envs["silero-vad"] == {}
    runtime.gpu_other_gib = 0.0
    cat.settings.release_linger_s = 0.0
    cat.scheduler.release(r.id)
    step(cat)
    run_unleased(cat, "kokoro-tts")  # custom: AISVC_VRAM_GB, capped at util_limit
    assert runtime.envs["kokoro-tts"] == {UTIL_VAR: "0.930", VRAM_VAR: "29.61"}


def test_without_a_measurement_a_gpu_model_gets_util_limit(cat, runtime):
    run_unleased(cat, "whisper-large-v3")
    assert runtime.envs["whisper-large-v3"] == {UTIL_VAR: "0.930", VRAM_VAR: "29.76"}  # 0.93 x gpu_total_gb 32
    assert cat.scheduler.runs["whisper-large-v3"].given == pytest.approx(0.93)


def test_a_model_waits_rather_than_loading_partly(cat, runtime):
    s = cat.scheduler
    cat.settings.gpu_free_wait_s = 0.5  # one 1 s retry of the start's wait for memory, not the default 30 s
    runtime.gpu_total_gib = 31.84
    runtime.gpu_other_gib = 20.0
    r = s.request("bench:job-1", ["qwen2-5-14b-instruct-awq"])
    step(cat)
    r = s.get(r.id)
    assert r.state == "queued" and "to load" in r.waiting_for and "outside the platform" in r.waiting_for
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
    assert r.state == "queued" and "waits for bge-m3 (starting)" in r.waiting_for
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
    assert r.state == "queued" and "needs at least 4.8 GiB of the GPU to load" in r.waiting_for and runtime.started == []
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
