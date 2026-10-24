"""Helpers for catalog tests: drive the scheduler by hand, and make audio and versions for the Dataset Catalog."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import time
import wave
import zipfile

from catalog.db import iso


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


def jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def version_zip(files: list[tuple[str, bytes]], refs: dict[str, str] | None = None, *, label_types=("asr",),
                format_version: int = 2, name: str = "Calls", audio: bool = False, top: str = "pkg",
                **manifest) -> bytes:
    """A zip of a version folder (or, with `audio`, a reference package that carries its audio)."""
    rows = "".join(f"{p},{sha(d)},0.5,16000\n" for p, d in files)
    m = {"format": "concord-reference-package", "format_version": format_version, "name": name,
         "created_at": "2026-10-24T09:00:00Z", "label_types": list(label_types),
         "counts": {"files": len(files), "segments": 0, "hours": 0.0},
         "audio_included": "all" if audio else "none", **manifest}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{top}/manifest.json", json.dumps(m))
        z.writestr(f"{top}/files.csv", "path,sha256,duration,sample_rate\n" + rows)
        for fname, text in (refs or {}).items():
            z.writestr(f"{top}/refs/{fname}", text)
        if audio:
            for p, d in files:
                z.writestr(f"{top}/audio/{p}", d)
    return buf.getvalue()


def new_dataset(api, name: str = "Calls", **kw) -> str:
    r = api.post("/api/datasets", json={"name": name, "app": "concord-label", **kw})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def publish(api, did: str, data: bytes, parent=None, app: str = "concord-label", force: bool = False):
    form = {"app": app, "force": str(force).lower()}
    if parent is not None:
        form["parent"] = str(parent)
    return api.post(f"/api/datasets/{did}/versions", files={"file": ("v.zip", data, "application/zip")}, data=form)


def age(api, sha256: str, hours: float = 25) -> None:
    """Make a stored file look older, for clean-up."""
    api.dc.db.execute("UPDATE audio SET created_at = ? WHERE sha256 = ?", (iso(time.time() - hours * 3600), sha256))


ASR = {"segments.jsonl": jsonl([{"file": "a.wav", "start": 0.0, "end": 0.2, "text": "hello there"},
                                {"file": "b.wav", "start": 0.1, "end": 0.4, "text": "good morning", "attributes": {"noise": "street"}}])}
