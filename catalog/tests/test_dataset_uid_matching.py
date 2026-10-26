"""uids.jsonl gives each uid to one segment: the nearest one within 0.01 s that has no uid yet, as Concord Bench and
Concord Label match them. Two speakers talking at the same time keep their own uids."""

import json
from pathlib import Path

from catalog_helpers import jsonl, sha, wav
from fastapi.testclient import TestClient

from aicore.datasets import DatasetCatalog, FakeDatasetCatalog

from catalog.app import create_app
from catalog.datasets import refs

A = wav(1)


def _version(root: Path, segs: list[dict], uids: list[dict]) -> Path:
    (root / "refs").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "concord-reference-package", "format_version": 2, "name": "Calls",
        "created_at": "2026-10-24T09:00:00Z", "label_types": ["asr", "diar"],
        "counts": {"files": 1, "segments": len(segs)}, "audio_included": "none"}))
    (root / "files.csv").write_text(f"path,sha256,duration,sample_rate\na.wav,{sha(A)},2.0,16000\n")
    (root / "refs" / "segments.seglst.json").write_text(json.dumps(
        [{"session_id": "a", "speaker": spk, "start_time": s, "end_time": e, "words": w} for spk, s, e, w in segs]))
    (root / "refs" / "uids.jsonl").write_text(jsonl([{"file": "a.wav", "start": s, "end": e, "uid": u}
                                                     for s, e, u in uids]))
    return root


def _uids(root: Path) -> list[tuple[str, str | None]]:
    return [(s["speaker"], s["uid"]) for s in refs.parse(root)["segments"]["a.wav"]]


def test_a_uid_goes_to_the_nearest_segment_without_one(tmp_path):
    # Both segments are within 0.01 s of each uid; u2 is nearer the second, and u1 then takes the first.
    root = _version(tmp_path / "v", [("a", 0.0, 1.0, "x"), ("b", 0.008, 1.0, "y")],
                    [(0.008, 1.0, "u2"), (0.0, 1.0, "u1"), (0.004, 1.0, "u3")])
    assert _uids(root) == [("a", "u1"), ("b", "u2")]


def test_the_service_and_the_fake_show_the_same_uids(cat, tmp_path):
    root = _version(tmp_path / "v", [("agent", 0.0, 1.0, "yes"), ("caller", 0.0, 1.0, "no")],
                    [(0.0, 1.0, "u-agent"), (0.0, 1.0, "u-caller")])
    (tmp_path / "a.wav").write_bytes(A)
    app = create_app(cat, start=False)
    with TestClient(app) as http:
        server = DatasetCatalog("http://testserver", app="concord-label", store=cat.settings.dataset_store, http=http)
        for dc in (server, FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")):
            dc.upload_audio(tmp_path / "a.wav")
            did = dc.create_dataset("Calls")["id"]
            dc.publish(did, root)
            assert _uids(dc.version_dir(did, 1)) == [("agent", "u-agent"), ("caller", "u-caller")]
            if dc is server:
                got = http.get(f"/api/datasets/{did}/versions/1/refs", params={"path": "a.wav"}).json()
                assert [s["uid"] for s in got["segments"]] == ["u-agent", "u-caller"]
    app.state.datasets.db.close()
