"""Unpacking a zip of audio runs ffprobe once per file: the probe that decides whether to keep a file is the one
stored with it."""

import io
import zipfile

import pytest
from catalog_helpers import sha, upload, wav
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


def test_unpack_probes_each_file_once_and_stores_that_probe(api):
    store = api.dc.store
    a, b, c = wav(50), wav(51), wav(52)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("calls/a.wav", a)
        z.writestr("calls/b.wav", b)
        z.writestr("calls/c.wav", c)
        z.writestr("calls/broken.wav", b"not audio at all")
    rec = upload(api, buf.getvalue(), "zip")
    upload(api, c)  # one the store has already
    probed = []

    def probe(path):
        data = path.read_bytes()
        probed.append(sha(data))
        if data == b"not audio at all":
            return {"duration": None, "sample_rate": None, "channels": None}
        return {"duration": 0.5, "sample_rate": 8000 + len(probed), "channels": 2}

    store.ffprobe = "ffprobe"
    store.probe = probe
    out = api.post("/api/store/unpack", json={"sha256": rec["sha256"]}).json()
    assert sorted(probed) == sorted([sha(a), sha(b), sha(c), sha(b"not audio at all")])
    assert out["skipped"] == [{"path": "calls/broken.wav", "error": "ffprobe can't read it"}]
    got = {f["path"]: (f["sha256"], f["sample_rate"], f["channels"]) for f in out["files"]}
    assert got["calls/a.wav"] == (sha(a), 8001, 2) and got["calls/b.wav"] == (sha(b), 8002, 2)
    assert got["calls/c.wav"][0] == sha(c) and got["calls/c.wav"][2] != 2  # stored before: its record stays
    assert store.record(sha(a))["sample_rate"] == 8001


def test_without_ffprobe_unpack_stores_every_audio_file_unprobed(api):
    store = api.dc.store
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.wav", wav(53))
        z.writestr("broken.wav", b"not audio")
    rec = upload(api, buf.getvalue(), "zip")
    store.ffprobe = None
    out = api.post("/api/store/unpack", json={"sha256": rec["sha256"]}).json()
    assert [(f["path"], f["duration"]) for f in out["files"]] == [("a.wav", None), ("broken.wav", None)]
    assert out["skipped"] == []
