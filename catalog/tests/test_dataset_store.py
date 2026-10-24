"""The Dataset Catalog's audio store (catalog/datasets/store.py), through the HTTP API on a temporary store."""

import io
import shutil
import stat
import zipfile

import pytest
from catalog_helpers import sha, upload, wav
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.settings import Settings

needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
A, B, C = wav(1), wav(2), wav(3)


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


# ----------------------------------------------------------------------------- settings

def test_store_location_comes_from_dataset_store_or_ai_data(monkeypatch, tmp_path):
    monkeypatch.delenv("DATASET_STORE", raising=False)
    monkeypatch.setenv("AI_DATA", str(tmp_path / "data"))
    assert Settings().dataset_store == tmp_path / "data" / "datasets"
    monkeypatch.setenv("DATASET_STORE", str(tmp_path / "elsewhere"))
    assert Settings().dataset_store == tmp_path / "elsewhere"
    monkeypatch.delenv("AI_DATA")
    monkeypatch.delenv("DATASET_STORE")
    assert Settings().dataset_store == Settings().dataset_store.home() / "ai-data" / "datasets"


# ----------------------------------------------------------------------------- audio store

@needs_ffprobe
def test_upload_resumes_and_commit_checks_and_probes(api, tmp_path):
    s = sha(A)
    assert api.post("/api/store/missing", json={"sha256": [s, s]}).json() == {"missing": [s]}
    assert api.put(f"/api/store/audio/{s}", params={"ext": "WAV", "offset": 0}, content=A[:1000]).json() == {"size": 1000}
    assert api.post("/api/store/missing", json={"sha256": [s]}).json() == {"missing": [s]}  # partial counts as missing
    wrong = api.put(f"/api/store/audio/{s}", params={"offset": 0}, content=A[:10])
    assert wrong.status_code == 409 and wrong.json()["error"]["size"] == 1000
    assert api.put(f"/api/store/audio/{s}", params={"offset": 1000}, content=A[1000:]).json() == {"size": len(A)}
    rec = api.post(f"/api/store/audio/{s}/commit").json()
    assert rec == {**rec, "sha256": s, "size": len(A), "ext": ".wav", "state": "ready", "sample_rate": 16000,
                   "channels": 1, "duration": 0.5}
    path = tmp_path / "datasets" / "audio" / s[:2] / s
    assert path.read_bytes() == A and stat.S_IMODE(path.stat().st_mode) == 0o644
    assert not (tmp_path / "datasets" / "tmp" / s).exists()
    assert api.post("/api/store/missing", json={"sha256": [s]}).json() == {"missing": []}
    got = api.get(f"/api/store/audio/{s}")
    assert got.content == A and got.headers["content-type"] == "audio/wav"
    # Already stored: a PUT writes nothing and a commit returns the same record.
    assert api.put(f"/api/store/audio/{s}", params={"offset": 0}, content=b"x").json() == {"size": len(A)}
    assert api.post(f"/api/store/audio/{s}/commit").json() == rec


def test_commit_with_the_wrong_bytes_is_422_and_removes_the_upload(api, tmp_path):
    s = sha(A)
    api.put(f"/api/store/audio/{s}", params={"offset": 0}, content=B)
    r = api.post(f"/api/store/audio/{s}/commit")
    assert r.status_code == 422 and r.json()["error"]["actual"] == sha(B)
    assert not (tmp_path / "datasets" / "tmp" / s).exists()
    assert api.post("/api/store/missing", json={"sha256": [s]}).json() == {"missing": [s]}
    assert api.put(f"/api/store/audio/{s}", params={"offset": 0}, content=A).json() == {"size": len(A)}
    assert api.post(f"/api/store/audio/{s}/commit").json()["state"] == "ready"
    assert api.post("/api/store/missing", json={"sha256": ["nope"]}).status_code == 422
    assert api.get(f"/api/store/audio/{sha(C)}").status_code == 404
    assert api.post(f"/api/store/audio/{sha(C)}/commit").status_code == 404


def test_holds_count_per_app(api):
    r = api.post("/api/store/holds", json={"app": "concord-bench", "add": [sha(A), sha(B), sha(A)]})
    assert r.json() == {"held": 2}
    assert api.post("/api/store/holds", json={"app": "concord-label", "add": [sha(A)]}).json() == {"held": 1}
    assert api.post("/api/store/holds", json={"app": "concord-bench", "remove": [sha(A)]}).json() == {"held": 1}
    assert api.post("/api/store/holds", json={"app": "x", "add": ["bad"]}).status_code == 422
    upload(api, A)
    holds = api.get("/api/store/stats").json()["holds"]
    assert holds == [{"app": "concord-bench", "files": 1, "bytes": 0}, {"app": "concord-label", "files": 1, "bytes": len(A)}]


@needs_ffprobe
def test_unpack_stores_and_probes_the_audio_of_a_zip(api, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("calls/a.wav", A)
        z.writestr("calls/sub/b.WAV", B)
        z.writestr("calls/notes.txt", "hi")
        z.writestr("calls/broken.wav", b"not audio at all")
        z.writestr("__MACOSX/calls/._a.wav", b"junk")
    rec = upload(api, buf.getvalue(), "zip")
    assert rec["duration"] is None
    upload(api, A)  # one file the store has already
    out = api.post("/api/store/unpack", json={"sha256": rec["sha256"]}).json()
    assert [(f["path"], f["sha256"], f["sample_rate"]) for f in out["files"]] == [
        ("calls/a.wav", sha(A), 16000), ("calls/sub/b.WAV", sha(B), 16000)]
    assert out["files"][1]["ext"] == ".wav"
    assert out["skipped"] == [{"path": "calls/notes.txt", "error": "not an audio file"},
                              {"path": "calls/broken.wav", "error": "ffprobe can't read it"}]
    assert api.post("/api/store/missing", json={"sha256": [rec["sha256"], sha(B)]}).json() == {"missing": [rec["sha256"]]}
    assert list((tmp_path / "datasets" / "tmp").iterdir()) == []
    assert api.post("/api/store/unpack", json={"sha256": sha(A)}).status_code == 422  # not a zip
