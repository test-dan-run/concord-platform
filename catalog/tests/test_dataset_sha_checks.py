"""A sha256 is exactly 64 lowercase hex characters: one with a newline after it is refused by the catalog's store
routes, by files.csv's check and by FakeDatasetCatalog alike."""

import contextlib
import json
from collections.abc import Iterator

import pytest
from catalog_helpers import sha, upload, wav
from fastapi.testclient import TestClient

from aicore.datasets import DatasetCatalog, DatasetCatalogError, FakeDatasetCatalog, InvalidVersion

from catalog.app import create_app

A = wav(1)
NEWLINE = "a" * 64 + "\n"


@contextlib.contextmanager
def _app(cat) -> Iterator[TestClient]:
    app = create_app(cat, start=False)
    with TestClient(app) as http:
        yield http
    app.state.datasets.db.close()


@pytest.fixture(params=["fake", "real"])
def dc(request, tmp_path):
    if request.param == "fake":
        yield FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")
    else:
        cat = request.getfixturevalue("cat")
        with _app(cat) as http:
            yield DatasetCatalog("http://testserver", app="concord-label", store=cat.settings.dataset_store, http=http)


def test_the_store_routes_refuse_a_sha256_with_a_newline(cat):
    with _app(cat) as api:
        upload(api, A)
        r = api.post("/api/store/holds", json={"app": "a", "add": [NEWLINE]})
        assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request"
        assert api.post("/api/store/holds", json={"app": "a", "remove": [sha(A) + "\n"]}).status_code == 422
        assert api.post("/api/store/missing", json={"sha256": [NEWLINE]}).status_code == 422
        assert api.get("/api/store/audio/" + sha(A) + "%0A").status_code == 422
        assert api.post("/api/store/holds", json={"app": "a", "add": [sha(A)]}).json() == {"held": 1}


def test_holds_and_missing_refuse_a_sha256_with_a_newline(dc):
    for call in (dc.hold, dc.release, dc.missing):
        with pytest.raises(DatasetCatalogError) as exc:
            call([NEWLINE])
        assert exc.value.status == 422
    assert dc.hold([]) == 0


def test_files_csv_refuses_a_sha256_with_a_newline(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    (tmp_path / "a.wav").write_bytes(A)
    dc.upload_audio(tmp_path / "a.wav")
    root = tmp_path / "v"
    (root / "refs").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "concord-reference-package", "format_version": 2, "name": "Calls",
        "created_at": "2026-10-24T09:00:00Z", "label_types": ["asr"], "counts": {"files": 1, "segments": 0},
        "audio_included": "none"}))
    (root / "files.csv").write_text(f'path,sha256,duration,sample_rate\na.wav,"{sha(A)}\n",0.5,16000\n')
    (root / "refs" / "segments.jsonl").write_text('{"file": "a.wav", "start": 0, "end": 0.2, "text": "hi"}\n')
    with pytest.raises(InvalidVersion) as exc:
        dc.publish(did, root)
    assert "files.csv: sha256 of a.wav" in exc.value.problems
