"""Helpers for catalog tests: drive the scheduler by hand, and make audio and versions for the Dataset Catalog."""

from __future__ import annotations

import hashlib
import io
import math
import struct
import time
import wave


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


# ----------------------------------------------------------------------------- Dataset Catalog

def wav(seed: int, seconds: float = 0.5, rate: int = 16000, channels: int = 1) -> bytes:
    """A short WAV whose bytes differ per seed."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i * (seed + 1) / 20))) * channels
                               for i in range(n)))
    return buf.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upload(api, data: bytes, ext: str = "wav") -> dict:
    """Upload and commit one file through the store's API, in two chunks."""
    s = sha(data)
    half = len(data) // 2
    assert api.put(f"/api/store/audio/{s}", params={"ext": ext, "offset": 0}, content=data[:half]).status_code == 200
    assert api.put(f"/api/store/audio/{s}", params={"ext": ext, "offset": half}, content=data[half:]).status_code == 200
    r = api.post(f"/api/store/audio/{s}/commit")
    assert r.status_code == 200, r.text
    return r.json()

