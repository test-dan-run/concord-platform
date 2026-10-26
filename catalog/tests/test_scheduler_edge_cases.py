"""Edge cases of the scheduler and the weights hasher: starts and stops that go wrong, and the starter thread
finishing while a tick runs."""

import sqlite3
import subprocess
import threading
import time

import pytest
from catalog_helpers import drain, step

from catalog.builds import WeightsHasher
from catalog.db import DB


def until(cond, timeout: float = 5.0) -> None:
    """Wait for a background thread to make `cond` true."""
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_start_that_crashes_fails_the_lease_and_frees_the_gpu(cat, runtime):
    s = cat.scheduler
    real = runtime.start

    def start(p, svc, env=None):
        if svc.name == "whisper-large-v3":
            raise FileNotFoundError("[Errno 2] No such file or directory: 'docker'")  # not a StartError
        return real(p, svc, env)

    runtime.start = start
    r = s.request("bench:job-1", ["whisper-large-v3"])
    drain(cat)
    step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "start crashed: [Errno 2]" in r.error
    assert not s.runs["whisper-large-v3"].starting and s.model_view("whisper-large-v3")["state"] == "failed"
    other = s.request("bench:job-2", ["bge-m3"])  # another GPU model does not wait for it
    step(cat)
    assert s.get(other.id).state == "active"


def test_the_weights_hasher_keeps_going_after_an_error(tmp_path, monkeypatch):
    db = DB(tmp_path / "catalog.db")
    real = db.execute
    failed = []

    def execute(sql, args=()):
        if sql.startswith("INSERT INTO weights_hash") and not failed:
            failed.append(sql)
            raise sqlite3.OperationalError("database is locked")  # not an OSError
        return real(sql, args)

    monkeypatch.setattr(db, "execute", execute)
    h = WeightsHasher(db)
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "model.safetensors").write_bytes(d.name.encode() * 10)
    assert h.get(a) is None
    until(lambda: failed and not h.hashing(a))
    until(lambda: h.get(b) is not None)  # the next folder is still hashed
    assert h._thread.is_alive()
    db.close()


def test_docker_runtime_stop_raises_when_compose_fails(monkeypatch):
    from catalog.runtime import DockerRuntime
    from tools import stack

    monkeypatch.setattr(stack, "compose", lambda p, *args, **kw: subprocess.CompletedProcess(args, 1, "", "no such service\n"))
    with pytest.raises(RuntimeError, match="docker compose stop x exited 1: no such service"):
        DockerRuntime().stop(None, ["x"])


# A model being stopped is down: a request for it waits for a new start, queued behind the stop.

def spun_down(cat, model: str):
    """A model whose lease ended and whose stop is queued but not yet run by the starter."""
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    r = s.request("bench:job-1", [model])
    step(cat)
    s.release(r.id)
    s.tick()
    assert s.runs[model].stopping
    return s.runs[model]


def test_the_gateway_waits_for_a_model_being_stopped_to_start_again(cat, runtime):
    s = cat.scheduler
    spun_down(cat, "whisper-large-v3")
    req = s.ensure_started("whisper-large-v3")
    assert req.state == "starting" and s.ensure_started("whisper-large-v3") is req  # no new request each poll
    drain(cat)
    step(cat)
    assert s.get(req.id).state == "done" and s.is_up("whisper-large-v3")


def test_a_model_being_stopped_must_find_its_memory_free_to_start_again(cat, runtime):
    s = cat.scheduler
    runtime.gpu_total_gib = 31.84
    spun_down(cat, "whisper-large-v3")
    runtime.gpu_other_gib = 28.0  # another process takes memory while the stop waits
    s.tick()
    r = s.request("bench:job-2", ["whisper-large-v3"])
    assert r.state == "queued" and "whisper-large-v3 needs 5.8 GiB free on the GPU to load" in r.waiting_for
    assert ("start", "whisper-large-v3") not in list(s._jobs.queue)
    runtime.gpu_other_gib = 0.0
    drain(cat)
    step(cat)
    assert s.get(r.id).state == "active"


# The tick lists containers before it takes the lock: the starter can finish a start or stop in between.

def tick_with_stale_listing(cat, runtime, meanwhile=None) -> None:
    """One tick whose container listing is taken just before the starter runs its queued jobs."""
    real = runtime.containers

    def containers(p):
        listing = real(p)
        runtime.containers = real
        drain(cat)  # the starter finishes while the tick is outside the lock
        if meanwhile:
            meanwhile()
        return listing

    runtime.containers = containers
    cat.scheduler.tick()
    assert runtime.containers is real


def test_a_stale_listing_keeps_what_a_start_was_given(cat, runtime):
    s = cat.scheduler
    r = s.request("bench:job-1", ["whisper-large-v3"])
    tick_with_stale_listing(cat, runtime)  # listed before the start finished: no container yet
    run = s.runs["whisper-large-v3"]
    assert run.running and run.given is not None and run.batch is not None
    step(cat)
    assert run.given is not None and run.batch is not None and not run.pinned
    assert s.get(r.id).state == "active" and runtime.started == ["whisper-large-v3"]


def test_a_stale_listing_does_not_start_a_restarted_model_twice(cat, runtime):
    s = cat.scheduler
    r = s.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    runtime.running.pop("whisper-large-v3")  # crashed
    s.tick()  # the lease queues a restart
    # Listed before the restart finished; then another request makes the lease active again.
    tick_with_stale_listing(cat, runtime, lambda: s.request("bench:job-2", ["silero-vad"]))
    assert ("start", "whisper-large-v3") not in list(s._jobs.queue) and s.runs["whisper-large-v3"].running
    step(cat)
    assert s.get(r.id).state == "active" and runtime.started.count("whisper-large-v3") == 2


def test_a_stale_listing_does_not_bring_back_a_stopped_model(cat, runtime):
    s = cat.scheduler
    run = spun_down(cat, "whisper-large-v3")
    tick_with_stale_listing(cat, runtime)  # listed before the stop finished: still running
    assert not s.is_up("whisper-large-v3") and not run.running and not run.pinned
    runtime.running["whisper-large-v3"] = 0.0  # then started outside the catalog: still picked up
    s.tick()
    assert s.is_up("whisper-large-v3") and run.pinned


def test_a_model_is_not_granted_memory_its_start_would_refuse(cat, runtime):
    s = cat.scheduler
    cat.settings.gpu_free_wait_s = 0.0
    runtime.gpu_total_gib = 31.84
    need = 0.15 * 31.84 + 1.0  # whisper-large-v3: gpu.util 0.15, plus 1 GiB left free
    runtime.gpu_other_gib = 31.84 - (need + 0.005)  # free: 5 MiB over that, but the start rounds its share down
    s.tick()
    r = s.request("bench:job-1", ["whisper-large-v3"])
    s.tick()
    r = s.get(r.id)
    assert r.state == "queued" and ("start", "whisper-large-v3") not in list(s._jobs.queue)
    assert "whisper-large-v3 needs 5.8 GiB free on the GPU to load" in r.waiting_for
    runtime.gpu_other_gib = 31.84 - (need + 0.05)
    step(cat)
    assert s.get(r.id).state == "active" and runtime.started == ["whisper-large-v3"]


def test_one_thread_reloads_changed_manifests_and_the_others_reuse_it(cat, monkeypatch):
    import catalog.scheduler as scheduler_module

    s = cat.scheduler
    s.platform()
    real = scheduler_module.load_platform
    loads, sigs = [], []
    go = threading.Event()

    def load_platform(root):
        loads.append(root)
        go.wait(10)  # the reload is slow: the other threads ask meanwhile
        return real(root)

    def manifest_sig():
        sigs.append(1)
        return ("changed",)  # a manifest changed

    monkeypatch.setattr(scheduler_module, "load_platform", load_platform)
    monkeypatch.setattr(s, "_manifest_sig", manifest_sig)
    got = []
    threads = [threading.Thread(target=lambda: got.append(s.platform())) for _ in range(6)]
    for t in threads:
        t.start()
    until(lambda: len(sigs) >= 6)  # every thread has seen the change
    go.set()
    for t in threads:
        t.join(10)
    assert len(got) == 6 and len({id(p) for p in got}) == 1
    assert len(loads) == 1


def test_a_lease_that_names_a_model_twice_names_it_once(cat):
    from fastapi.testclient import TestClient

    from catalog.app import create_app

    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as api:
        r = api.post("/api/leases", json={"holder": "bench:job-1", "models": ["whisper-large-v3", "default-asr", "silero-vad"]})
        assert r.status_code == 201, r.text
        assert r.json()["models"] == ["whisper-large-v3", "silero-vad"]  # an alias and its id are one model; order kept
        r = api.post("/api/leases", json={"holder": "bench:job-2", "models": ["silero-vad", "nope", "nope"]})
        assert r.status_code == 404 and r.json()["error"]["message"].startswith("unknown service 'nope'")
