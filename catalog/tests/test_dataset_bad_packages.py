"""Malformed version folders and packages are refused with 422 invalid_version and their problems, never a 500:
check_folder reads the references with the same parse that publishing uses, so a folder it passes can be stored.
An import that fails after making its dataset removes that dataset again."""

import io
import json
import zipfile

import pytest
from catalog_helpers import ASR, jsonl, new_dataset, publish, upload, version_zip, wav
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.datasets.versions import Versions

A, B = wav(1), wav(2)
FILES = [("a.wav", A), ("b.wav", B)]


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        upload(c, A)
        upload(c, B)
        yield c
    app.state.datasets.db.close()


def _replace(data: bytes, name: str, content: bytes) -> bytes:
    """The zip with one member's bytes replaced."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, content if n == name else src.read(n))
    return out.getvalue()


def _uid(**rec) -> dict:
    return {**ASR, "uids.jsonl": jsonl([{"file": "a.wav", "start": 0.0, "end": 0.2, "uid": "u", **rec}])}


BAD = {
    "label_types holds a list": version_zip(FILES, ASR, label_types=(["asr"],)),
    "files.csv is not UTF-8": _replace(version_zip(FILES, ASR), "pkg/files.csv",
                                       b"path,sha256,duration,sample_rate\n\xff\xfe,x,1,1\n"),
    "uids.jsonl is not UTF-8": version_zip(FILES, {**ASR, "uids.jsonl": b"\xff\xfe\n"}),
    "a uid's start is a string": version_zip(FILES, _uid(start="x")),
    "a uid's file is a list": version_zip(FILES, _uid(file=["a.wav"])),
    "attributes files is a list": version_zip(FILES, {**ASR, "attributes.json": json.dumps({"files": ["a.wav"]})}),
    "a SegLST session_id is a number": version_zip(FILES, {"segments.seglst.json": json.dumps(
        [{"session_id": 7, "speaker": "s", "start_time": 0, "end_time": 1, "words": "hi"}])}),
    "name is a number": version_zip(FILES, ASR, name=123),
    "description is an object": version_zip(FILES, ASR, description={"x": 1}),
    "tags is a string": version_zip(FILES, ASR, tags="x"),
    "a tag is a number": version_zip(FILES, ASR, tags=["x", 1]),
}


@pytest.mark.parametrize("case", BAD)
def test_a_malformed_version_is_422_not_500(api, case):
    did = new_dataset(api)
    r = publish(api, did, BAD[case])
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version", r.text
    assert r.json()["error"]["problems"]
    assert api.get(f"/api/datasets/{did}").json()["versions"] == []


# The import checks with the same check_folder: one manifest problem and one references file that can't be read.
@pytest.mark.parametrize("case", ["a tag is a number", "a SegLST session_id is a number"])
def test_a_malformed_package_previews_as_not_ok_and_imports_nothing(api, case):
    files = {"file": ("p.zip", BAD[case], "application/zip")}
    preview = api.post("/api/datasets/import", files=files, data={"preview": "true"})
    assert preview.status_code == 200 and preview.json()["ok"] is False and preview.json()["problems"], preview.text
    r = api.post("/api/datasets/import", files=files)
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version", r.text
    assert api.get("/api/datasets").json() == []


def test_the_new_problems_say_what_is_wrong(api):
    did = new_dataset(api)
    problems = lambda case: publish(api, did, BAD[case]).json()["error"]["problems"]  # noqa: E731
    assert "manifest.json: label_types [['asr']]" in problems("label_types holds a list")
    assert "refs/uids.jsonl line 1: needs file (a files.csv path), start, end, uid" in problems("a uid's start is a string")
    assert "manifest.json: name must be a string" in problems("name is a number")
    assert "manifest.json: tags must be a list of strings" in problems("a tag is a number")
    assert "refs/attributes.json: files must be an object" in problems("attributes files is a list")
    assert any(p.startswith("files.csv is not UTF-8") for p in problems("files.csv is not UTF-8"))
    assert any(p.startswith("refs/segments.seglst.json: can't be read") for p in problems("a SegLST session_id is a number"))


def test_attributes_json_is_the_references_file_in_format_version_1(api):
    """Version 1 has no labels: its attributes.json is read as its references file, as check_folder counts it."""
    did = new_dataset(api)
    labels = {"attributes.json": json.dumps({"files": {"a.wav": {"cond": "x"}}})}
    r = publish(api, did, version_zip(FILES, labels, label_types=("diar",), format_version=1))
    assert r.status_code == 201, r.text
    assert r.json()["labels"] == []
    refs = api.get(f"/api/datasets/{did}/versions/1/refs", params={"path": "a.wav"}).json()
    assert (refs["format"], refs["segments"], refs["labels"]) == ("attributes.json", None, {})
    empty = version_zip(FILES, {"attributes.json": "[]"}, label_types=("diar",), format_version=1)
    r = api.post("/api/datasets/import", files={"file": ("p.zip", empty, "application/zip")})
    assert r.status_code == 201, r.text
    assert api.get(f"/api/datasets/{r.json()['dataset_id']}/versions/1/files").json()["files"][0]["segments"] == 0


def test_an_import_that_fails_after_making_its_dataset_removes_it(api, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Versions, "_write_folder", fail)
    r = api.post("/api/datasets/import", files={"file": ("p.zip", version_zip(FILES, ASR), "application/zip")})
    assert r.status_code == 500
    assert api.get("/api/datasets").json() == []
