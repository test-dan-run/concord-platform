"""A use is stored with its app and ref stripped, so removing it with the same spaces around them must find it, on
the catalog service and on FakeDatasetCatalog alike."""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from catalog_helpers import ASR, new_dataset, publish, sha, upload, version_zip, wav
from fastapi.testclient import TestClient

from aicore.datasets import DatasetCatalog, DatasetCatalogError, FakeDatasetCatalog

from catalog.app import create_app

A = wav(1)


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


def _version(dc, tmp_path: Path) -> str:
    """A dataset with version 1 holding a.wav."""
    did = dc.create_dataset("Calls")["id"]
    (tmp_path / "a.wav").write_bytes(A)
    dc.upload_audio(tmp_path / "a.wav")
    root = tmp_path / "v"
    (root / "refs").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "concord-reference-package", "format_version": 2, "name": "Calls",
        "created_at": "2026-10-24T09:00:00Z", "label_types": ["asr"], "counts": {"files": 1, "segments": 0},
        "audio_included": "none"}))
    (root / "files.csv").write_text(f"path,sha256,duration,sample_rate\na.wav,{sha(A)},0.5,16000\n")
    (root / "refs" / "segments.jsonl").write_text(ASR["segments.jsonl"])
    dc.publish(did, root)
    return did


def test_a_use_with_spaces_around_its_ref_can_be_removed(dc, tmp_path):
    did = _version(dc, tmp_path)
    dc.add_use(did, 1, " 7 ")
    assert [u["ref"] for u in dc.version(did, 1)["uses"]] == ["7"]
    dc.remove_use(did, 1, " 7 ")
    assert dc.version(did, 1)["uses"] == []
    with pytest.raises(DatasetCatalogError) as exc:
        dc.add_use(did, 1, " ")
    assert exc.value.status == 422 and dc.version(did, 1)["uses"] == []


def test_the_uses_route_strips_app_and_ref_on_delete_too(cat):
    with _app(cat) as api:
        did = new_dataset(api)
        upload(api, A)
        assert publish(api, did, version_zip([("a.wav", A)], ASR)).status_code == 201
        uses = f"/api/datasets/{did}/versions/1/uses"
        assert api.post(uses, json={"app": " concord-bench ", "ref": " 7 "}).status_code == 204
        assert [(u["app"], u["ref"]) for u in api.get(f"/api/datasets/{did}/versions/1").json()["uses"]] == [
            ("concord-bench", "7")]
        assert api.delete(uses, params={"app": " concord-bench ", "ref": " 7 "}).status_code == 204
        assert api.get(f"/api/datasets/{did}/versions/1").json()["uses"] == []
