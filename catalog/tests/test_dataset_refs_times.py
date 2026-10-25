"""Reference segment times are finite numbers: a NaN or an infinite start, end or duration in a references file, or
in refs/uids.jsonl, is a problem on the catalog service and on FakeDatasetCatalog alike. Stored, a NaN start came
back as null and broke the comparison of versions, which sorts segments by start."""

import json

import pytest
from catalog_helpers import new_dataset, publish, sha, upload, version_zip, wav
from fastapi.testclient import TestClient

from aicore.datasets import FakeDatasetCatalog, InvalidVersion

from catalog.app import create_app

A, B = wav(1), wav(2)
FILES = [("a.wav", A), ("b.wav", B)]
GOOD = '{"file": "b.wav", "start": 0.1, "end": 0.4, "text": "good"}\n'


def times(name: str, start: str, end: str) -> str:
    return f"refs/{name}: a segment of a.wav has start {start} and end {end}; times must be finite numbers"


# (refs files, label types, the problem)
BAD = [
    ({"segments.jsonl": '{"file": "a.wav", "start": NaN, "end": 0.2, "text": "hello"}\n' + GOOD}, ("asr",),
     times("segments.jsonl", "nan", "0.2")),
    ({"segments.jsonl": '{"file": "a.wav", "start": 0.1, "end": Infinity, "text": "hello"}\n' + GOOD}, ("asr",),
     times("segments.jsonl", "0.1", "inf")),
    ({"segments.jsonl": '{"file": "a.wav", "start": -Infinity, "end": 0.2, "text": "hello"}\n' + GOOD}, ("asr",),
     times("segments.jsonl", "-inf", "0.2")),
    ({"segments.jsonl": '{"file": "a.wav", "start": "nan", "end": 0.2, "text": "hello"}\n' + GOOD}, ("asr",),
     times("segments.jsonl", "nan", "0.2")),
    ({"segments.seglst.json": '[{"session_id": "a", "speaker": "s1", "start_time": 0.1, "end_time": 1e400, '
                              '"words": "hello"}]'}, ("asr", "diar"), times("segments.seglst.json", "0.1", "inf")),
    ({"refs.rttm": "SPEAKER a 1 nan 0.5 <NA> <NA> s1 <NA> <NA>\n"}, ("diar",), times("refs.rttm", "nan", "nan")),
    ({"refs.rttm": "SPEAKER a 1 0.1 inf <NA> <NA> s1 <NA> <NA>\n"}, ("diar",), times("refs.rttm", "0.1", "inf")),
    ({"segments.jsonl": GOOD, "uids.jsonl": '{"file": "b.wav", "start": NaN, "end": 0.4, "uid": "u1"}\n'}, ("asr",),
     "refs/uids.jsonl line 1: needs file (a files.csv path), start, end, uid"),
    ({"segments.jsonl": GOOD, "uids.jsonl": '{"file": "b.wav", "start": 0.1, "end": Infinity, "uid": "u1"}\n'},
     ("asr",), "refs/uids.jsonl line 1: needs file (a files.csv path), start, end, uid"),
    ({"segments.jsonl": GOOD, "uids.jsonl": '{"file": "b.wav", "start": 1' + "0" * 400 + ', "end": 0.4, "uid": "u"}\n'},
     ("asr",), "refs/uids.jsonl line 1: needs file (a files.csv path), start, end, uid"),
    ({"segments.jsonl": GOOD, "uids.jsonl": '{"file": "b.wav", "start": true, "end": 0.4, "uid": "u1"}\n'}, ("asr",),
     "refs/uids.jsonl line 1: needs file (a files.csv path), start, end, uid"),
]
IDS = ["jsonl-nan", "jsonl-inf", "jsonl-minus-inf", "jsonl-nan-text", "seglst-1e400", "rttm-nan", "rttm-inf",
       "uids-nan", "uids-inf", "uids-too-large", "uids-true"]


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        upload(c, A)
        upload(c, B)
        yield c
    app.state.datasets.db.close()


@pytest.mark.parametrize(("refs", "label_types", "problem"), BAD, ids=IDS)
def test_a_time_that_is_not_finite_is_422(api, refs, label_types, problem):
    did = new_dataset(api)
    r = publish(api, did, version_zip(FILES, refs, label_types=label_types))
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version", r.text
    assert r.json()["error"]["problems"] == [problem]
    assert api.get(f"/api/datasets/{did}").json()["versions"] == []


def test_finite_times_stay_valid(api):
    refs = {"segments.jsonl": '{"file": "a.wav", "start": 0, "end": 1e3, "text": "hello"}\n' + GOOD,
            "uids.jsonl": '{"file": "a.wav", "start": 0, "end": 1000, "uid": "u1"}\n'}
    did = new_dataset(api)
    r = publish(api, did, version_zip(FILES, refs))
    assert r.status_code == 201, r.text
    g = api.get(f"/api/datasets/{did}/versions/1/refs", params={"path": "a.wav"}).json()
    assert [(s["start"], s["end"], s["uid"]) for s in g["segments"]] == [(0.0, 1000.0, "u1")]


@pytest.mark.parametrize(("refs", "label_types", "problem"), BAD, ids=IDS)
def test_the_fake_refuses_the_same_times(tmp_path, refs, label_types, problem):
    fake = FakeDatasetCatalog(tmp_path / "store", app="concord-label")
    for data in (A, B):
        fake.add_audio(data)
    did = fake.create_dataset("Calls")["id"]
    root = tmp_path / "v"
    (root / "refs").mkdir(parents=True)
    m = {"format": "concord-reference-package", "format_version": 2, "name": "Calls",
         "created_at": "2026-10-24T09:00:00Z", "label_types": list(label_types),
         "counts": {"files": 2, "segments": 0, "hours": 0.0}, "audio_included": "none"}
    (root / "manifest.json").write_text(json.dumps(m))
    (root / "files.csv").write_text("path,sha256,duration,sample_rate\n"
                                    + "".join(f"{p},{sha(d)},0.5,16000\n" for p, d in FILES))
    for name, text in refs.items():
        (root / "refs" / name).write_text(text)
    with pytest.raises(InvalidVersion) as exc:
        fake.publish(did, root)
    assert problem in str(exc.value)
    assert fake.dataset(did)["versions"] == []
