"""A speaker in a references file is a string (or missing). The apps read every speaker as a string and
attributes.json names speakers by string, so a version whose SegLST or jsonl gives a speaker as a list, a number or
true is refused, where it used to be stored and then answered 500 on GET .../files."""

import io
import json
import zipfile

import pytest
from catalog_helpers import jsonl, new_dataset, publish, upload, version_zip, wav
from fastapi.testclient import TestClient

from aicore.dataset_format import check_folder
from catalog.app import create_app

A, B = wav(60), wav(61)


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


def seglst(speakers):
    return {"segments.seglst.json": json.dumps([
        {"session_id": "a", "speaker": s, "start_time": i, "end_time": i + 0.5, "words": "w"}
        for i, s in enumerate(speakers)])}


CASES = [
    (seglst([["x"], "y"]), "refs/segments.seglst.json: speaker ['x'] of a.wav is not a string"),
    (seglst(["s", 1]), "refs/segments.seglst.json: speaker 1 of a.wav is not a string"),
    (seglst([True]), "refs/segments.seglst.json: speaker True of a.wav is not a string"),
    ({"segments.jsonl": jsonl([{"file": "a.wav", "start": 0, "end": 1, "text": "hi", "speaker": {"id": 1}}])},
     "refs/segments.jsonl: speaker {'id': 1} of a.wav is not a string"),
]


@pytest.mark.parametrize("refs,problem", CASES)
def test_a_speaker_that_is_not_a_string_is_refused(api, tmp_path, refs, problem):
    did = new_dataset(api)
    upload(api, A)
    types = ("asr", "diar") if "segments.seglst.json" in refs else ("asr",)
    data = version_zip([("a.wav", A)], refs, label_types=types)
    r = publish(api, did, data)
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version"
    assert r.json()["error"]["problems"] == [problem]
    # FakeDatasetCatalog checks a folder with the same code, so it refuses it with the same problem.
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp_path)
    assert check_folder(tmp_path / "pkg") == [problem]


def test_string_and_missing_speakers_are_stored_and_listed(api):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    refs = {"segments.seglst.json": json.dumps(
        [{"session_id": "a", "speaker": s, "start_time": i, "end_time": i + 0.5, "words": "w"}
         for i, s in enumerate(["b", None, "a", "", "b"])])}
    r = publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], refs, label_types=("asr", "diar")))
    assert r.status_code == 201, r.text
    files = api.get(f"/api/datasets/{did}/versions/1/files")
    assert files.status_code == 200
    assert {f["path"]: f["speakers"] for f in files.json()["files"]} == {"a.wav": ["", "a", "b"], "b.wav": []}
