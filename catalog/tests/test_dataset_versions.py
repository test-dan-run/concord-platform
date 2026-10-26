"""The Dataset Catalog's datasets and versions (catalog/datasets/versions.py and refs.py): the version format, the
publishing rules, uses, import, audio-only versions, compare and package download, through the HTTP API on a
temporary store."""

import io
import json
import queue
import shutil
import zipfile
from pathlib import Path

import pytest
from catalog_helpers import ASR, jsonl, new_dataset, publish, sha, upload, version_zip, wav
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.datasets.versions import check_folder, check_samples

A, B, C = wav(1), wav(2), wav(3)


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


@pytest.fixture
def events(cat):
    q = cat.events.subscribe()
    yield q
    cat.events.unsubscribe(q)


def drain(q: queue.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ----------------------------------------------------------------------------- the version format

def test_the_contract_samples_are_valid_versions():
    assert check_samples() == []


def test_the_format_rules_for_a_version_folder(tmp_path):
    samples = Path(__file__).resolve().parents[2] / "contracts" / "datasets" / "samples"
    shutil.copytree(samples, tmp_path / "s")
    s = tmp_path / "s"
    (s / "version-audio-only" / "refs").mkdir()
    (s / "version-audio-only" / "refs" / "segments.jsonl").write_text("")
    shutil.copy(s / "version-labelled" / "refs" / "attributes.json", s / "version-diar-v1" / "refs")
    (s / "version-asr" / "audio").mkdir()
    m = json.loads((s / "version-labelled" / "manifest.json").read_text())
    (s / "version-labelled" / "manifest.json").write_text(json.dumps({**m, "label_types": []}))
    problems = check_samples(s)
    assert "version-audio-only: refs/ must be empty when label_types is empty (a version with audio only)" in problems
    assert any(p.startswith("version-diar-v1: refs/ must hold exactly one references file besides uids.jsonl") for p in problems)
    assert "version-asr: audio/ is not part of a version" in problems
    assert "version-labelled: manifest.json: label_types []" in problems


# ----------------------------------------------------------------------------- publishing (§5.3)

def test_deleting_versions_keeps_numbers_unused_and_marks_parents_deleted(api, tmp_path, events):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    publish(api, did, version_zip([("a.wav", A)], ASR))
    assert drain(events)[-1] == ("dataset", {"id": did, "n": 1, "action": "published"})
    publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], ASR), parent=1)
    api.post(f"/api/datasets/{did}/versions/2/uses", json={"app": "concord-bench", "ref": "12"})
    assert api.delete(f"/api/datasets/{did}/versions/2").status_code == 204
    assert ("dataset", {"id": did, "n": 2, "action": "deleted"}) in drain(events)
    assert not (tmp_path / "datasets" / "versions" / did / "2").exists()
    assert api.get(f"/api/datasets/{did}/versions/2").status_code == 404
    assert api.dc.db.all("SELECT * FROM version_file WHERE n = 2") == [] and api.dc.db.all("SELECT * FROM use") == []
    # A publish based on the deleted version is stale; the latest is v1 again.
    r = publish(api, did, version_zip([("b.wav", B)], ASR), parent=2)
    assert r.status_code == 409 and "v2 was deleted" in r.json()["error"]["message"] and r.json()["error"]["latest"]["n"] == 1
    v3 = publish(api, did, version_zip([("b.wav", B)], ASR), parent=2, force=True).json()
    assert v3["n"] == 3 and v3["parent_n"] == 2 and v3["parent_deleted"] is True
    # Deleting the latest doesn't free its number either.
    api.delete(f"/api/datasets/{did}/versions/3")
    assert publish(api, did, version_zip([("b.wav", B)], ASR), parent=1).json()["n"] == 4
    # The audio stays in the store.
    assert api.post("/api/store/missing", json={"sha256": [sha(A), sha(B)]}).json() == {"missing": []}
    assert api.delete(f"/api/datasets/{did}/versions/3").status_code == 404


def test_deleting_a_dataset_removes_all_its_versions(api, tmp_path, events):
    did = new_dataset(api)
    upload(api, A)
    publish(api, did, version_zip([("a.wav", A)], ASR))
    api.post(f"/api/datasets/{did}/versions/1/uses", json={"app": "concord-bench", "ref": "12"})
    assert api.delete(f"/api/datasets/{did}").status_code == 204
    assert drain(events)[-1] == ("dataset", {"id": did, "n": None, "action": "deleted"})
    assert api.get(f"/api/datasets/{did}").status_code == 404
    assert not (tmp_path / "datasets" / "versions" / did).exists()
    for table in ("version", "version_file", "use", "dataset"):
        assert api.dc.db.all(f"SELECT * FROM {table}") == []
    assert api.post("/api/store/missing", json={"sha256": [sha(A)]}).json() == {"missing": []}


def test_stems_must_be_unique(api):
    did = new_dataset(api)
    r = publish(api, did, version_zip([("a.wav", A), ("x/A.flac", B)], ASR))
    assert "stem of x/A.flac repeats a.wav" in r.json()["error"]["message"]


# ----------------------------------------------------------------------------- import

def test_import_a_package_with_its_audio_as_a_new_dataset_then_a_new_version(api, events):
    pkg = version_zip([("a.wav", A), ("b.wav", B)], ASR, audio=True, name="Help desk", language="en", tags=["x"])
    files = {"file": ("help.zip", pkg, "application/zip")}
    preview = api.post("/api/datasets/import", files=files, data={"preview": "true"}).json()
    assert preview["ok"] and preview["files"] == 2 and preview["missing"] == [] and preview["manifest"]["name"] == "Help desk"
    assert api.get("/api/datasets").json() == []  # a preview stores nothing
    r = api.post("/api/datasets/import", files=files, data={"app": "concord-bench"})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["n"] == 1 and v["published_by"] == "concord-bench" and v["manifest"]["audio_included"] == "none"
    d = api.get(f"/api/datasets/{v['dataset_id']}").json()
    assert (d["name"], d["language"], d["tags"], d["created_by"]) == ("Help desk", "en", ["x"], "concord-bench")
    assert api.post("/api/store/missing", json={"sha256": [sha(A), sha(B)]}).json() == {"missing": []}
    # The same zip into the same dataset is the same version; a changed one becomes v2 on top of the latest.
    again = api.post("/api/datasets/import", files=files, data={"dataset_id": d["id"]})
    assert again.status_code == 200 and again.json()["n"] == 1
    assert api.post("/api/datasets/import", files=files, data={"dataset_id": d["id"], "preview": "1"}).json()["same_as_latest"]
    pkg2 = version_zip([("a.wav", A)], ASR, name="Help desk")
    r2 = api.post("/api/datasets/import", files={"file": ("h.zip", pkg2, "application/zip")}, data={"dataset_id": d["id"]})
    assert r2.status_code == 201 and r2.json()["n"] == 2 and r2.json()["parent_n"] == 1


def test_import_fails_without_audio_and_on_bad_audio_and_creates_no_dataset(api):
    no_audio = version_zip([("c.wav", C)], ASR)
    r = api.post("/api/datasets/import", files={"file": ("c.zip", no_audio, "application/zip")})
    assert r.status_code == 422 and r.json()["error"]["missing"] == [sha(C)]
    preview = api.post("/api/datasets/import", files={"file": ("c.zip", no_audio, "application/zip")},
                       data={"preview": "true"}).json()
    assert preview["missing"] == [sha(C)]
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as z, zipfile.ZipFile(io.BytesIO(version_zip([("a.wav", A)], ASR, audio=True))) as src:
        for name in src.namelist():
            z.writestr(name, B if name.endswith("a.wav") else src.read(name))
    r = api.post("/api/datasets/import", files={"file": ("a.zip", bad.getvalue(), "application/zip")})
    assert r.status_code == 422 and r.json()["error"]["type"] == "hash_mismatch"
    assert api.post("/api/datasets/import", files={"file": ("x.zip", b"not a zip", "application/zip")}).status_code == 422
    handmade = io.BytesIO()
    with zipfile.ZipFile(handmade, "w") as z:
        z.writestr("audio/a.wav", A)
        z.writestr("a.stm", "a 1 spk 0 1 hi\n")
    r = api.post("/api/datasets/import", files={"file": ("h.zip", handmade.getvalue(), "application/zip")})
    assert r.status_code == 422 and "annotated upload" in r.json()["error"]["message"]
    assert api.get("/api/datasets").json() == []


# ----------------------------------------------------------------------------- reading versions

LABELLED = {
    "segments.seglst.json": json.dumps([
        {"session_id": "a", "speaker": "agent", "start_time": 0.0, "end_time": 0.2, "words": "thank you for calling",
         "attributes": {"noise": "none"}},
        {"session_id": "a", "speaker": "caller", "start_time": 0.25, "end_time": 0.45, "words": "hi there"},
        {"session_id": "b", "speaker": "agent", "start_time": 0.1, "end_time": 0.3, "words": "hello"}]),
    "uids.jsonl": jsonl([{"file": "a.wav", "start": 0.0, "end": 0.2, "uid": "u1"},
                         {"file": "a.wav", "start": 0.25, "end": 0.45, "uid": "u2"}]),
    "attributes.json": json.dumps({"files": {"a.wav": {"condition": "clean"}},
                                   "speakers": {"a.wav": {"agent": {"gender": "F", "accent": {"region": "south"}}}}}),
}


def test_files_are_paged_with_segment_counts_and_labels_and_refs_per_file(api):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    v = publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], LABELLED, label_types=("asr", "diar"))).json()
    assert v["labels"] == ["condition", "noise", "speaker.accent.region", "speaker.gender"]
    page = api.get(f"/api/datasets/{did}/versions/1/files", params={"offset": 1, "limit": 1}).json()
    assert page["total"] == 2 and [f["path"] for f in page["files"]] == ["b.wav"]
    first = api.get(f"/api/datasets/{did}/versions/1/files").json()["files"][0]
    assert first == {"path": "a.wav", "sha256": sha(A), "duration": 0.5, "sample_rate": 16000, "segments": 2,
                     "speakers": ["agent", "caller"], "labels": {"condition": "clean"}}
    refs = api.get(f"/api/datasets/{did}/versions/1/refs", params={"path": "a.wav"}).json()
    assert refs["format"] == "segments.seglst.json" and refs["labels"] == {"condition": "clean"}
    assert refs["speakers"] == {"agent": {"gender": "F", "accent.region": "south"}}
    assert refs["segments"][0] == {"start": 0.0, "end": 0.2, "speaker": "agent", "text": "thank you for calling",
                                   "uid": "u1", "labels": {"noise": "none"}}
    assert api.get(f"/api/datasets/{did}/versions/1/refs", params={"path": "zz.wav"}).status_code == 404


def test_compare_lists_file_segment_and_label_changes(api):
    did = new_dataset(api)
    for d in (A, B, C):
        upload(api, d)
    publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], LABELLED, label_types=("asr", "diar")))
    segs = json.loads(LABELLED["segments.seglst.json"])
    segs[0]["words"] = "thanks for calling"           # u1: text edited
    segs[1]["start_time"] = 0.3                        # u2: times edited, matched by uid
    segs.append({"session_id": "a", "speaker": "agent", "start_time": 0.46, "end_time": 0.49, "words": "bye"})
    segs = [s for s in segs if s["session_id"] != "b"] + [
        {"session_id": "c", "speaker": "agent", "start_time": 0.0, "end_time": 0.1, "words": "new"}]
    v2 = {**LABELLED, "segments.seglst.json": json.dumps(segs),
          "uids.jsonl": jsonl([{"file": "a.wav", "start": 0.0, "end": 0.2, "uid": "u1"},
                               {"file": "a.wav", "start": 0.3, "end": 0.45, "uid": "u2"}]),
          "attributes.json": json.dumps({"files": {"a.wav": {"condition": "noisy", "site": "x"}}})}
    publish(api, did, version_zip([("a.wav", A), ("c.wav", C)], v2, label_types=("asr", "diar")), parent=1)
    cmp = api.get(f"/api/datasets/{did}/compare", params={"a": 1, "b": 2}).json()
    assert cmp["files"] == {"added": ["c.wav"], "removed": ["b.wav"], "same": 0}
    assert cmp["segments_compared"] is True
    [a] = cmp["changed"]
    assert a["path"] == "a.wav" and "audio_changed" not in a
    assert [s["text"] for s in a["segments"]["added"]] == ["bye"] and a["segments"]["removed"] == []
    edited = {e["b"]["uid"]: e for e in a["segments"]["edited"]}
    assert edited["u1"]["diff"] == [{"op": "replace", "a": "thank you", "b": "thanks"},
                                    {"op": "equal", "a": "for calling", "b": "for calling"}]
    assert edited["u2"]["a"]["start"] == 0.25 and edited["u2"]["b"]["start"] == 0.3
    assert a["labels"]["b"]["files"] == {"condition": "noisy", "site": "x"}
    assert cmp["labels"] == {"added": ["site"], "removed": ["speaker.accent.region", "speaker.gender"]}
    assert api.get(f"/api/datasets/{did}/compare", params={"a": 1, "b": 9}).status_code == 404


def test_compare_without_uids_matches_by_times_and_overlap(api):
    did = new_dataset(api)
    upload(api, A)
    rttm1 = "SPEAKER a 1 0.000 0.200 <NA> <NA> s1 <NA> <NA>\nSPEAKER a 1 0.300 0.100 <NA> <NA> s2 <NA> <NA>\n"
    rttm2 = "SPEAKER a 1 0.000 0.200 <NA> <NA> s1 <NA> <NA>\nSPEAKER a 1 0.310 0.100 <NA> <NA> s1 <NA> <NA>\n"
    publish(api, did, version_zip([("a.wav", A)], {"refs.rttm": rttm1}, label_types=("diar",), format_version=1))
    publish(api, did, version_zip([("a.wav", A)], {"refs.rttm": rttm2}, label_types=("diar",), format_version=1), parent=1)
    [a] = api.get(f"/api/datasets/{did}/compare", params={"a": 1, "b": 2}).json()["changed"]
    assert a["segments"]["added"] == [] and a["segments"]["removed"] == []
    [e] = a["segments"]["edited"]
    assert (e["a"]["speaker"], e["b"]["speaker"], e["b"]["start"]) == ("s2", "s1", 0.31)


def test_package_download_with_and_without_audio(api, tmp_path):
    did = new_dataset(api, name="Help desk / calls")
    upload(api, A)
    upload(api, B)
    publish(api, did, version_zip([("a.wav", A), ("sub/b.wav", B)], {**LABELLED, "attributes.json": "{}"},
                                  label_types=("asr", "diar")))
    for audio in (False, True):
        r = api.get(f"/api/datasets/{did}/versions/1/package", params={"audio": audio})
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert r.headers["content-disposition"] == 'attachment; filename="Help-desk-calls-v1.zip"'
        out = tmp_path / f"pkg-{audio}"
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(out)
        root = out / "Help-desk-calls-v1"
        assert check_folder(root, package=True) == []
        assert json.loads((root / "manifest.json").read_text())["audio_included"] == ("all" if audio else "none")
        assert (root / "audio" / "sub" / "b.wav").is_file() is audio
        if audio:
            assert (root / "audio" / "a.wav").read_bytes() == A
        assert (root / "refs" / "uids.jsonl").read_text() == LABELLED["uids.jsonl"]
        # A downloaded package publishes back as the same version.
        assert publish(api, did, r.content).status_code == 200


def test_datasets_are_listed_with_their_latest_version_and_filters(api):
    calls = new_dataset(api, "Help desk calls", language="en", description="phone support")
    meetings = new_dataset(api, "Meetings", language="ms", tags=["internal"])
    upload(api, A)
    publish(api, calls, version_zip([("a.wav", A)], ASR))
    publish(api, meetings, version_zip([("a.wav", A)], {"refs.rttm": "SPEAKER a 1 0 1 <NA> <NA> s <NA> <NA>\n"},
                                       label_types=("diar",)), app="concord-bench")
    listed = api.get("/api/datasets").json()
    assert {d["id"]: d["latest"]["n"] for d in listed} == {calls: 1, meetings: 1}
    ids = lambda **p: [d["id"] for d in api.get("/api/datasets", params=p).json()]  # noqa: E731
    assert ids(q="PHONE") == [calls] and ids(q="internal") == [meetings]
    assert ids(language="ms") == [meetings] and ids(label_type="asr") == [calls]
    assert ids(app="concord-bench") == [meetings]
    assert api.patch(f"/api/datasets/{calls}", json={"name": "Calls", "tags": ["x"]}).json()["name"] == "Calls"
    assert api.patch(f"/api/datasets/{calls}", json={"name": " "}).status_code == 422
    assert api.post("/api/datasets", json={"name": "x"}).status_code == 422  # app is required
