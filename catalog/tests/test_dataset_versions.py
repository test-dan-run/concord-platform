"""The Dataset Catalog's datasets and versions (catalog/datasets/versions.py and refs.py): the version format, the
publishing rules, uses, import, audio-only versions, compare and package download, through the HTTP API on a
temporary store."""

import io
import json
import queue
import shutil
import zipfile

import pytest
from catalog_helpers import ASR, jsonl, new_dataset, publish, sha, upload, version_zip, wav
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.datasets.versions import check_folder, digest

needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
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


# ----------------------------------------------------------------------------- publishing (§5.3)

def test_publish_fills_the_catalog_fields_and_writes_the_folder(api, tmp_path, events):
    did = new_dataset(api, language="en", tags=["b", "a", "b"])
    upload(api, A)
    upload(api, B)
    r = publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], ASR))
    assert r.status_code == 201, r.text
    v = r.json()
    assert v == {**v, "dataset_id": did, "n": 1, "parent_n": None, "parent_deleted": False, "format_version": 2,
                 "label_types": ["asr"], "labels": ["noise"], "published_by": "concord-label", "uses": []}
    m = v["manifest"]
    assert m["source"] == {"app": "concord-label", "install_id": "dataset-catalog", "dataset_uid": did,
                           "dataset_name": "Calls"}
    assert m["export_version"] == 1 and m["audio_included"] == "none" and m["created_at"] == v["published_at"]
    folder = tmp_path / "datasets" / "versions" / did / "1"
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) == [
        "files.csv", "manifest.json", "refs", "refs/segments.jsonl"]
    assert check_folder(folder) == [] and digest(folder) == v["digest"]
    assert json.loads((folder / "manifest.json").read_text())["source"]["dataset_uid"] == did
    assert drain(events)[-1] == ("dataset", {"id": did, "n": 1, "action": "published"})
    d = api.get(f"/api/datasets/{did}").json()
    assert d["tags"] == ["a", "b"] and d["latest"]["n"] == 1 and [x["n"] for x in d["versions"]] == [1]
    assert api.get(f"/api/datasets/{did}/versions/1").json() == v


def test_the_same_content_again_returns_the_latest_version_with_200(api):
    did = new_dataset(api)
    upload(api, A)
    first = publish(api, did, version_zip([("a.wav", A)], ASR)).json()
    # The catalog's own fields don't count, nor does the zip's layout: a double click, or the same folder written again.
    again = publish(api, did, version_zip([("a.wav", A)], ASR, created_at="2030-01-01T00:00:00Z", top="other",
                                          source={"app": "x", "install_id": "y", "dataset_uid": "z", "dataset_name": "w"},
                                          export_version=7))
    assert again.status_code == 200 and again.json()["n"] == 1 and again.json()["digest"] == first["digest"]
    changed = publish(api, did, version_zip([("a.wav", A)], ASR, description="new"), parent=1)
    assert changed.status_code == 201 and changed.json()["n"] == 2 and changed.json()["parent_n"] == 1
    # Same content as v1, which is no longer the latest: a new version.
    back = publish(api, did, version_zip([("a.wav", A)], ASR), parent=2)
    assert back.status_code == 201 and back.json()["n"] == 3 and back.json()["digest"] == first["digest"]


def test_publish_with_audio_missing_from_the_store_is_422(api):
    did = new_dataset(api)
    upload(api, A)
    r = publish(api, did, version_zip([("a.wav", A), ("b.wav", B), ("c.wav", C)], ASR))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["type"] == "missing_audio" and err["missing"] == [sha(B), sha(C)]
    assert api.get(f"/api/datasets/{did}").json()["versions"] == []


def test_a_stale_parent_is_409_unless_forced(api):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    publish(api, did, version_zip([("a.wav", A)], ASR))
    v2 = publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], ASR), parent=1).json()
    stale = publish(api, did, version_zip([("b.wav", B)], ASR), parent=1)
    assert stale.status_code == 409
    err = stale.json()["error"]
    assert err["type"] == "stale_parent" and err["latest"]["n"] == 2 and err["latest"]["published_by"] == "concord-label"
    assert "latest version is v2" in err["message"]
    assert publish(api, did, version_zip([("b.wav", B)], ASR)).status_code == 409  # no parent, but v2 exists
    forced = publish(api, did, version_zip([("b.wav", B)], ASR), parent=1, force=True)
    assert forced.status_code == 201 and forced.json()["n"] == 3 and forced.json()["parent_n"] == 1
    assert v2["n"] == 2
    assert publish(api, did, version_zip([("c.wav", C)], ASR), parent=9).status_code == 422  # never existed
    assert publish(api, "f" * 32, version_zip([("a.wav", A)], ASR)).status_code == 404


def test_deleting_versions_keeps_numbers_unused_and_marks_parents_deleted(api, tmp_path, events):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    publish(api, did, version_zip([("a.wav", A)], ASR))
    publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], ASR), parent=1)
    assert api.delete(f"/api/datasets/{did}/versions/2").status_code == 204
    assert ("dataset", {"id": did, "n": 2, "action": "deleted"}) in drain(events)
    assert not (tmp_path / "datasets" / "versions" / did / "2").exists()
    assert api.get(f"/api/datasets/{did}/versions/2").status_code == 404
    assert api.dc.db.all("SELECT * FROM version_file WHERE n = 2") == []
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


@pytest.mark.parametrize(("zip_kw", "problem"), [
    ({"refs": {"segments.jsonl": "", "refs.rttm": ""}}, "exactly one references file"),
    ({"refs": {}}, "exactly one references file"),
    ({"refs": ASR, "label_types": ()}, "label_types []"),
    ({"refs": ASR, "format_version": 4}, "format_version 4"),
    ({"refs": {**ASR, "attributes.json": json.dumps({"files": {"z.wav": {"x": 1}}})}}, "files.z.wav is not a files.csv path"),
    ({"refs": {}, "label_types": (), "format_version": 3, "audio_included": "all"}, "audio_included is 'all'"),
])
def test_an_invalid_version_is_422_with_its_problems(api, zip_kw, problem):
    did = new_dataset(api)
    upload(api, A)
    upload(api, B)
    kw = {"refs": None, **zip_kw}
    r = publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], kw.pop("refs"), **kw))
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version"
    assert any(problem in p for p in r.json()["error"]["problems"]), r.json()


def test_stems_must_be_unique(api):
    did = new_dataset(api)
    r = publish(api, did, version_zip([("a.wav", A), ("x/A.flac", B)], ASR))
    assert "stem of x/A.flac repeats a.wav" in r.json()["error"]["message"]


# ----------------------------------------------------------------------------- uses

def test_uses_are_listed_on_the_version_and_go_with_it(api):
    did = new_dataset(api)
    upload(api, A)
    publish(api, did, version_zip([("a.wav", A)], ASR))
    for ref in ("12", "12", "13"):
        assert api.post(f"/api/datasets/{did}/versions/1/uses", json={"app": "concord-bench", "ref": ref}).status_code == 204
    assert [(u["app"], u["ref"]) for u in api.get(f"/api/datasets/{did}/versions/1").json()["uses"]] == [
        ("concord-bench", "12"), ("concord-bench", "13")]
    assert api.delete(f"/api/datasets/{did}/versions/1/uses", params={"app": "concord-bench", "ref": "13"}).status_code == 204
    assert len(api.get(f"/api/datasets/{did}/versions/1").json()["uses"]) == 1
    assert api.post(f"/api/datasets/{did}/versions/5/uses", json={"app": "a", "ref": "1"}).status_code == 404
    api.delete(f"/api/datasets/{did}/versions/1")
    assert api.dc.db.all("SELECT * FROM use") == []
    # An app releasing its use after the version was deleted still gets 204.
    assert api.delete(f"/api/datasets/{did}/versions/1/uses", params={"app": "concord-bench", "ref": "12"}).status_code == 204


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


# ----------------------------------------------------------------------------- audio-only versions (§5.2, §6.1)

@needs_ffprobe
def test_audio_only_versions_add_files_until_references_arrive(api, tmp_path):
    did = new_dataset(api, language="en")
    for d in (A, B, C):
        upload(api, d)
    r = api.post(f"/api/datasets/{did}/versions/audio", json={"files": [{"path": "a.wav", "sha256": sha(A)}]})
    assert r.status_code == 201, r.text
    v1 = r.json()
    assert v1["format_version"] == 3 and v1["label_types"] == [] and v1["counts"]["files"] == 1
    folder = tmp_path / "datasets" / "versions" / did / "1"
    assert not (folder / "refs").exists() and check_folder(folder) == []
    assert (folder / "files.csv").read_text() == f"path,sha256,duration,sample_rate\na.wav,{sha(A)},0.5,16000\n"
    body = {"files": [{"path": "a.wav", "sha256": sha(A)}, {"path": "sub/b.wav", "sha256": sha(B)}], "parent": 1,
            "app": "dataset-catalog"}
    v2 = api.post(f"/api/datasets/{did}/versions/audio", json=body).json()
    assert v2["n"] == 2 and v2["parent_n"] == 1 and v2["counts"]["files"] == 2
    assert api.post(f"/api/datasets/{did}/versions/audio", json=body).status_code == 200  # nothing new
    stale = api.post(f"/api/datasets/{did}/versions/audio", json={**body, "files": [{"path": "c.wav", "sha256": sha(C)}]})
    assert stale.status_code == 409 and stale.json()["error"]["type"] == "stale_parent"
    clash = {"files": [{"path": "a.wav", "sha256": sha(C)}], "parent": 2}
    assert api.post(f"/api/datasets/{did}/versions/audio", json=clash).status_code == 422
    missing = {"files": [{"path": "d.wav", "sha256": "e" * 64}], "parent": 2}
    assert api.post(f"/api/datasets/{did}/versions/audio", json=missing).json()["error"]["missing"] == ["e" * 64]
    # Once a version has references, new audio comes through an app.
    publish(api, did, version_zip([("a.wav", A), ("b.wav", B)], ASR), parent=2)
    r = api.post(f"/api/datasets/{did}/versions/audio", json={"files": [{"path": "c.wav", "sha256": sha(C)}], "parent": 3})
    assert r.status_code == 409 and r.json()["error"]["type"] == "has_references"


@needs_ffprobe
def test_audio_ffprobe_cant_read_is_left_out_of_audio_only_versions(api):
    did = new_dataset(api)
    rec = upload(api, b"plain text, not audio", "wav")
    r = api.post(f"/api/datasets/{did}/versions/audio", json={"files": [{"path": "t.wav", "sha256": rec["sha256"]}]})
    assert r.status_code == 422 and r.json()["error"]["unreadable"] == ["t.wav"]


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
