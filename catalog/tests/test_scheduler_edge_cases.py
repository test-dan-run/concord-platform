"""Edge cases of the scheduler and the weights hasher: starts and stops that go wrong, and the starter thread
finishing while a tick runs."""

import sqlite3
import time

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
