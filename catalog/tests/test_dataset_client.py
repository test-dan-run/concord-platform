"""aicore.datasets: every case runs against FakeDatasetCatalog and against DatasetCatalog talking to the catalog
service in process, so the fake keeps to the server's rules."""

import contextlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from catalog_helpers import ASR, jsonl, sha, wav
from fastapi.testclient import TestClient

from aicore import datasets as client_module
from aicore.datasets import (DatasetCatalog, DatasetCatalogError, FakeDatasetCatalog, HasReferences, InvalidVersion,
                             MissingAudio, NotFound, StaleParent, store_path)

from catalog.app import create_app
from catalog.datasets.versions import check_folder
from catalog.datasets.versions import digest as server_digest
from catalog.settings import Settings

A, B, C = wav(1), wav(2), wav(3)
LABELLED = {
    "segments.seglst.json": json.dumps([
        {"session_id": "a", "speaker": "agent", "start_time": 0.0, "end_time": 0.2, "words": "thank you",
         "attributes": {"noise": "none"}},
        {"session_id": "a", "speaker": "caller", "start_time": 0.25, "end_time": 0.45, "words": "hi"},
        {"session_id": "b", "speaker": "agent", "start_time": 0.1, "end_time": 0.3, "words": "hello"}]),
    "uids.jsonl": jsonl([{"file": "a.wav", "start": 0.0, "end": 0.2, "uid": "u1"}]),
    "attributes.json": json.dumps({"files": {"a.wav": {"condition": "clean"}},
                                   "speakers": {"a.wav": {"agent": {"accent": {"region": "south"}}}}}),
}


@contextlib.contextmanager
def real(cat) -> Iterator[DatasetCatalog]:
    """A DatasetCatalog on the catalog service's app, in process, with the store the service writes to."""
    app = create_app(cat, start=False)
    with TestClient(app) as http:
        yield DatasetCatalog("http://testserver", app="concord-label", store=cat.settings.dataset_store, http=http)
    app.state.datasets.db.close()


@pytest.fixture(params=["fake", "real"])
def dc(request, tmp_path):
    if request.param == "fake":
        yield FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")
    else:
        with real(request.getfixturevalue("cat")) as c:
            yield c


@pytest.fixture
def server(cat):
    with real(cat) as c:
        yield c


def files(tmp_path: Path, *items: tuple[str, bytes]) -> list[tuple[str, Path]]:
    """Audio files on disk, as an app has them."""
    out = []
    for name, data in items:
        p = tmp_path / "originals" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        out.append((name, p))
    return out


def upload(dc, tmp_path: Path, *items: tuple[str, bytes]) -> list[str]:
    return [dc.upload_audio(p) for _, p in files(tmp_path, *items)]


def folder(tmp_path: Path, items: list[tuple[str, bytes]], refs: dict[str, str] | None = None, *,
           label_types=("asr",), format_version: int = 2, audio: bool = False, name: str = "Calls",
           **manifest) -> Path:
    """A version folder, as an app writes one (with `audio`, a reference package with audio/)."""
    root = tmp_path / "out" / f"v{len(list((tmp_path / 'out').glob('*'))) if (tmp_path / 'out').exists() else 0}"
    (root / "refs").mkdir(parents=True)
    m = {"format": "concord-reference-package", "format_version": format_version, "name": name,
         "created_at": "2026-10-24T09:00:00Z", "label_types": list(label_types),
         "counts": {"files": len(items), "segments": 0, "hours": 0.0},
         "audio_included": "all" if audio else "none", **manifest}
    (root / "manifest.json").write_text(json.dumps(m))
    (root / "files.csv").write_text("path,sha256,duration,sample_rate\n" +
                                    "".join(f"{p},{sha(d)},0.5,16000\n" for p, d in items))
    for fname, text in (refs or {}).items():
        (root / "refs" / fname).write_text(text)
    for p, d in items if audio else []:
        (root / "audio" / p).parent.mkdir(parents=True, exist_ok=True)
        (root / "audio" / p).write_bytes(d)
    (root / "outputs").mkdir()
    (root / "outputs" / "seg.jsonl").write_text("{}\n")  # left out of a version
    return root


# ----------------------------------------------------------------------------- datasets

def test_datasets_are_created_listed_changed_and_deleted(dc, tmp_path):
    calls = dc.create_dataset("Help desk calls", language="en", description="phone support", tags=["b", "a", "b"])
    assert calls == {**calls, "name": "Help desk calls", "language": "en", "tags": ["a", "b"],
                     "created_by": "concord-label", "latest": None, "versions": []}
    assert len(calls["id"]) == 32
    meetings = dc.create_dataset("Meetings", language="ms", tags=["internal"])["id"]
    upload(dc, tmp_path, ("a.wav", A))
    dc.publish(calls["id"], folder(tmp_path, [("a.wav", A)], ASR))
    ids = lambda **kw: [d["id"] for d in dc.datasets(**kw)]  # noqa: E731
    assert set(ids()) == {calls["id"], meetings}
    assert ids(q="PHONE") == [calls["id"]] and ids(q="internal") == [meetings]
    assert ids(language="ms") == [meetings] and ids(label_type="asr") == [calls["id"]]
    assert set(ids(app="concord-label")) == {calls["id"], meetings} and ids(app="concord-bench") == []
    assert dc.datasets(label_type="asr")[0]["latest"]["n"] == 1
    changed = dc.update_dataset(calls["id"], name=" Calls ", tags=["x"])
    assert (changed["name"], changed["tags"], changed["description"]) == ("Calls", ["x"], "phone support")
    assert dc.update_dataset(calls["id"], description="")["description"] is None
    with pytest.raises(DatasetCatalogError) as exc:
        dc.update_dataset(calls["id"], name=" ")
    assert exc.value.status == 422
    with pytest.raises(DatasetCatalogError):
        dc.create_dataset(" ")
    dc.delete_dataset(calls["id"])
    assert ids() == [meetings] and not dc.version_dir(calls["id"], 1).exists()
    with pytest.raises(NotFound):
        dc.dataset(calls["id"])
    with pytest.raises(NotFound):
        dc.delete_dataset(calls["id"])


# ----------------------------------------------------------------------------- audio store

def test_upload_audio_stores_the_original_once_where_the_mount_has_it(dc, tmp_path):
    [(_, a)] = files(tmp_path, ("call.WAV", A))
    assert dc.missing([sha(A), sha(B), sha(A)]) == [sha(A), sha(B)]
    assert dc.upload_audio(a) == sha(A)
    assert dc.audio_path(sha(A)) == dc.store / "audio" / sha(A)[:2] / sha(A)
    assert dc.audio_path(sha(A)).read_bytes() == A
    assert dc.missing([sha(A), sha(B)]) == [sha(B)]
    assert dc.upload_audio(a, sha(A)) == sha(A)  # the store has it: nothing is sent
    [(_, b)] = files(tmp_path, ("b.wav", B))
    with pytest.raises(DatasetCatalogError) as exc:
        dc.upload_audio(b, sha(C))
    assert exc.value.status == 422 and exc.value.type == "hash_mismatch"
    assert dc.missing([sha(B), sha(C)]) == [sha(B), sha(C)]
    with pytest.raises(DatasetCatalogError) as exc:
        dc.missing(["nope"])
    assert exc.value.status == 422


def test_holds_count_per_app(dc):
    assert dc.hold([sha(A), sha(B), sha(A)]) == 2
    assert dc.hold([sha(B)]) == 2
    assert dc.release([sha(A), sha(C)]) == 1
    assert dc.hold([]) == 1 and dc.release([]) == 1
    with pytest.raises(DatasetCatalogError) as exc:
        dc.hold(["x" * 64])
    assert exc.value.status == 422


# ----------------------------------------------------------------------------- publishing (§5.3)

def test_publish_writes_the_version_where_the_mount_has_it(dc, tmp_path):
    did = dc.create_dataset("Calls", language="en")["id"]
    upload(dc, tmp_path, ("a.wav", A), ("b.wav", B))
    v = dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B)], LABELLED, label_types=("asr", "diar")))
    assert v == {**v, "dataset_id": did, "n": 1, "parent_n": None, "parent_deleted": False, "format_version": 2,
                 "label_types": ["asr", "diar"], "labels": ["condition", "noise", "speaker.accent.region"],
                 "counts": {"files": 2, "segments": 0, "hours": 0.0}, "published_by": "concord-label", "uses": [],
                 "created": True}
    m = v["manifest"]
    assert m["source"] == {"app": "concord-label", "install_id": "dataset-catalog", "dataset_uid": did,
                           "dataset_name": "Calls"}
    assert m["export_version"] == 1 and m["audio_included"] == "none" and m["created_at"] == v["published_at"]
    out = dc.version_dir(did, 1)
    assert out == dc.store / "versions" / did / "1"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == [
        "files.csv", "manifest.json", "refs", "refs/attributes.json", "refs/segments.seglst.json", "refs/uids.jsonl"]
    assert check_folder(out) == [] and server_digest(out) == v["digest"]
    assert json.loads((out / "manifest.json").read_text())["source"]["dataset_uid"] == did
    got = {k: x for k, x in v.items() if k != "created"}
    assert dc.version(did, 1) == got
    d = dc.dataset(did)
    assert d["latest"] == got and d["versions"] == [got]
    assert dc.files(did, 1) == [
        {"path": "a.wav", "sha256": sha(A), "duration": 0.5, "sample_rate": 16000, "segments": 2,
         "speakers": ["agent", "caller"], "labels": {"condition": "clean"}},
        {"path": "b.wav", "sha256": sha(B), "duration": 0.5, "sample_rate": 16000, "segments": 1,
         "speakers": ["agent"], "labels": {}}]
    with pytest.raises(NotFound):
        dc.version(did, 2)
    with pytest.raises(NotFound):
        dc.publish("f" * 32, folder(tmp_path, [("a.wav", A)], ASR))


def test_files_are_listed_across_pages(dc, tmp_path, monkeypatch):
    monkeypatch.setattr(client_module, "FILES_PAGE", 2)
    did = dc.create_dataset("Calls")["id"]
    items = [(f"f{i}.wav", wav(i)) for i in range(5)]
    upload(dc, tmp_path, *items)
    dc.publish(did, folder(tmp_path, items, {"refs.rttm": "SPEAKER f1 1 0 1 <NA> <NA> s1 <NA> <NA>\n"},
                           label_types=("diar",), format_version=1))
    listed = dc.files(did, 1)
    assert [f["path"] for f in listed] == [p for p, _ in items]
    assert [f["segments"] for f in listed] == [0, 1, 0, 0, 0] and listed[1]["speakers"] == ["s1"]


def test_the_same_content_again_returns_the_latest_version(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A))
    first = dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR))
    # The catalog's own fields don't count: a double click, or the same folder written again later.
    again = dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR, created_at="2030-01-01T00:00:00Z", export_version=7,
                                   source={"app": "x", "install_id": "y", "dataset_uid": "z", "dataset_name": "w"}))
    assert again["created"] is False and again["n"] == 1 and again["digest"] == first["digest"]
    changed = dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR, description="new"), parent=1)
    assert changed["created"] and changed["n"] == 2 and changed["parent_n"] == 1
    back = dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR), parent=2)  # v1's content, not the latest: new
    assert back["created"] and back["n"] == 3 and back["digest"] == first["digest"]


def test_publish_with_audio_missing_from_the_store_fails(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A))
    with pytest.raises(MissingAudio) as exc:
        dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B), ("c.wav", C)], ASR))
    assert exc.value.status == 422 and exc.value.missing == [sha(B), sha(C)]
    assert dc.dataset(did)["versions"] == []


def test_a_package_folder_brings_its_audio(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    v = dc.publish(did, folder(tmp_path, [("a.wav", A), ("sub/b.wav", B)], ASR, audio=True))
    assert v["n"] == 1 and v["manifest"]["audio_included"] == "none"
    assert dc.missing([sha(A), sha(B)]) == [] and dc.audio_path(sha(B)).read_bytes() == B
    assert not (dc.version_dir(did, 1) / "audio").exists()


def test_a_stale_parent_conflicts_unless_forced(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A), ("b.wav", B))
    dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR))
    dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B)], ASR), parent=1)
    with pytest.raises(StaleParent) as exc:
        dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR), parent=1)
    assert exc.value.status == 409 and "latest version is v2" in exc.value.message
    assert exc.value.latest["n"] == 2 and exc.value.latest["published_by"] == "concord-label"
    with pytest.raises(StaleParent):
        dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR))  # no parent, but v2 exists
    forced = dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR), parent=1, force=True)
    assert forced["n"] == 3 and forced["parent_n"] == 1
    with pytest.raises(DatasetCatalogError) as exc:
        dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR), parent=9)  # never existed
    assert exc.value.status == 422


def test_deleted_numbers_are_never_used_again_and_a_deleted_parent_is_stale(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A), ("b.wav", B))
    dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR))
    dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B)], ASR), parent=1)
    dc.delete_version(did, 2)
    assert not dc.version_dir(did, 2).exists() and dc.version_dir(did, 1).is_dir()
    with pytest.raises(NotFound):
        dc.version(did, 2)
    with pytest.raises(StaleParent) as exc:
        dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR), parent=2)
    assert exc.value.latest["n"] == 1
    v3 = dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR), parent=2, force=True)
    assert (v3["n"], v3["parent_n"], v3["parent_deleted"]) == (3, 2, True)
    dc.delete_version(did, 3)
    assert dc.publish(did, folder(tmp_path, [("b.wav", B)], ASR), parent=1)["n"] == 4
    assert dc.missing([sha(A), sha(B)]) == []  # the audio stays
    with pytest.raises(NotFound):
        dc.delete_version(did, 3)


@pytest.mark.parametrize(("kw", "problem"), [
    ({"refs": {"segments.jsonl": "", "refs.rttm": ""}}, "exactly one references file"),
    ({"refs": {}}, "exactly one references file"),
    ({"refs": ASR, "label_types": ()}, "label_types []"),
    ({"refs": ASR, "format_version": 4}, "format_version 4"),
    ({"refs": {}, "label_types": (), "format_version": 3, "audio_included": "all"}, "audio_included is 'all'"),
])
def test_an_invalid_version_lists_its_problems(dc, tmp_path, kw, problem):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A))
    kw = dict(kw)
    with pytest.raises(InvalidVersion) as exc:
        dc.publish(did, folder(tmp_path, [("a.wav", A)], kw.pop("refs"), **kw))
    assert exc.value.status == 422 and any(problem in p for p in exc.value.problems), exc.value.problems


def test_a_folder_without_a_manifest_is_invalid(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(InvalidVersion):
        dc.publish(did, tmp_path / "empty")


# ----------------------------------------------------------------------------- audio-only versions (§5.2)

@pytest.fixture
def probed(dc):
    if isinstance(dc, DatasetCatalog) and shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")
    return dc


def test_audio_only_versions_add_files_until_references_arrive(probed, tmp_path):
    dc = probed
    did = dc.create_dataset("Calls", language="en")["id"]
    upload(dc, tmp_path, ("a.wav", A), ("b.wav", B), ("c.wav", C))
    v1 = dc.publish_audio(did, {"a.wav": sha(A)})
    assert v1["created"] and v1["format_version"] == 3 and v1["label_types"] == [] and v1["counts"]["files"] == 1
    csv = (dc.version_dir(did, 1) / "files.csv").read_text()
    assert csv == f"path,sha256,duration,sample_rate\na.wav,{sha(A)},0.5,16000\n"
    assert not (dc.version_dir(did, 1) / "refs").exists() and check_folder(dc.version_dir(did, 1)) == []
    both = [{"path": "a.wav", "sha256": sha(A)}, {"path": "sub/b.wav", "sha256": sha(B)}]
    v2 = dc.publish_audio(did, both, parent=1)
    assert (v2["n"], v2["parent_n"], v2["counts"]["files"]) == (2, 1, 2)
    assert dc.publish_audio(did, both, parent=2)["created"] is False
    with pytest.raises(StaleParent):
        dc.publish_audio(did, {"c.wav": sha(C)}, parent=1)
    with pytest.raises(DatasetCatalogError) as exc:
        dc.publish_audio(did, {"a.wav": sha(C)}, parent=2)  # the path has other audio
    assert exc.value.status == 422
    with pytest.raises(MissingAudio) as exc:
        dc.publish_audio(did, {"d.wav": "e" * 64}, parent=2)
    assert exc.value.missing == ["e" * 64]
    dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B)], ASR), parent=2)
    with pytest.raises(HasReferences) as exc:
        dc.publish_audio(did, {"c.wav": sha(C)}, parent=3)
    assert exc.value.status == 409 and exc.value.latest["n"] == 3


def test_audio_that_cant_be_read_is_left_out_of_audio_only_versions(probed, tmp_path):
    dc = probed
    did = dc.create_dataset("Calls")["id"]
    [s] = upload(dc, tmp_path, ("t.wav", b"plain text, not audio"))
    with pytest.raises(DatasetCatalogError) as exc:
        dc.publish_audio(did, {"t.wav": s})
    assert exc.value.status == 422 and exc.value.extra["unreadable"] == ["t.wav"]


# ----------------------------------------------------------------------------- uses

def test_uses_are_listed_on_the_version_and_go_with_it(dc, tmp_path):
    did = dc.create_dataset("Calls")["id"]
    upload(dc, tmp_path, ("a.wav", A))
    dc.publish(did, folder(tmp_path, [("a.wav", A)], ASR))
    for ref in ("12", "12", 13):
        dc.add_use(did, 1, ref)
    uses = [(u["app"], u["ref"]) for u in dc.version(did, 1)["uses"]]
    assert uses == [("concord-label", "12"), ("concord-label", "13")]
    assert [d["id"] for d in dc.datasets(app="concord-label")] == [did]
    dc.remove_use(did, 1, "13")
    assert len(dc.version(did, 1)["uses"]) == 1
    with pytest.raises(NotFound):
        dc.add_use(did, 5, "1")
    dc.delete_version(did, 1)
    dc.remove_use(did, 1, "12")  # after the version was deleted: still fine


# ----------------------------------------------------------------------------- the fake and the server agree

def test_the_fake_and_the_server_write_the_same_versions(server, tmp_path):
    fake = FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")
    out = []
    for dc in (fake, server):
        upload(dc, tmp_path, ("a.wav", A), ("b.wav", B))
        did = dc.create_dataset("Calls", language="en", tags=["x"])["id"]
        v = dc.publish(did, folder(tmp_path, [("a.wav", A), ("b.wav", B)], LABELLED, label_types=("asr", "diar")))
        dc.add_use(did, 1, "7")
        out.append((did, dc.dataset(did), dc.files(did, 1), v))
    (fid, fd, ff, fv), (sid, sd, sf, sv) = out
    assert ff == sf and fv["digest"] == sv["digest"]

    def same(x, y, path=""):
        """Equal apart from ids and times."""
        assert type(x) is type(y), path
        if isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for k in x:
                if k not in ("id", "dataset_id", "dataset_uid", "created_at", "updated_at", "published_at"):
                    same(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list):
            assert len(x) == len(y), path
            for i, (a, b) in enumerate(zip(x, y, strict=True)):
                same(a, b, f"{path}[{i}]")
        else:
            assert x == y, path

    same(fd, sd)
    same(fv, sv)
    for name in ("files.csv", "refs/segments.seglst.json", "refs/uids.jsonl", "refs/attributes.json"):
        assert (fake.version_dir(fid, 1) / name).read_bytes() == (server.version_dir(sid, 1) / name).read_bytes()


# ----------------------------------------------------------------------------- the client alone

def test_upload_resumes_a_partial_upload_in_chunks(server, tmp_path, monkeypatch):
    monkeypatch.setattr(client_module, "CHUNK_BYTES", 4000)
    [(_, a)] = files(tmp_path, ("a.wav", A))
    s = sha(A)
    server._http.put(f"/api/store/audio/{s}", params={"ext": "wav", "offset": 0}, content=A[:1500])
    puts = []
    request = server._http.request

    def counting(method, url, **kw):
        if method == "PUT":
            puts.append((kw["params"]["offset"], len(kw["content"])))
        return request(method, url, **kw)

    monkeypatch.setattr(server._http, "request", counting)
    assert server.upload_audio(a) == s
    assert puts == [(0, 0), (1500, 4000), (5500, 4000), (9500, 4000), (13500, len(A) - 13500)]
    assert server.audio_path(s).read_bytes() == A and server.missing([s]) == []


def test_upload_goes_on_after_the_connection_drops(server, tmp_path, monkeypatch):
    monkeypatch.setattr(client_module, "CHUNK_BYTES", 4000)
    monkeypatch.setattr(client_module, "RETRY_S", 0)
    [(_, a)] = files(tmp_path, ("a.wav", A))
    request, dropped = server._http.request, []

    def drop_once(method, url, **kw):
        r = request(method, url, **kw)
        if method == "PUT" and kw["params"]["offset"] == 4000 and not dropped:
            dropped.append(1)
            raise httpx.ReadError("connection reset")  # the bytes arrived, the answer didn't
        return r

    monkeypatch.setattr(server._http, "request", drop_once)
    assert server.upload_audio(a) == sha(A) and dropped
    assert server.audio_path(sha(A)).read_bytes() == A


def test_errors_without_the_catalogs_shape_still_raise(server):
    with pytest.raises(DatasetCatalogError) as exc:
        server._req("POST", "/datasets", json={"name": "x"})  # FastAPI's own 422: {"detail": [...]}
    assert exc.value.status == 422
    with pytest.raises(NotFound):
        server._req("GET", "/no-such-route")


def test_the_store_defaults_match_the_service(monkeypatch, tmp_path):
    for env, expected in (({"AI_DATA": str(tmp_path / "data")}, tmp_path / "data" / "datasets"),
                          ({"DATASET_STORE": str(tmp_path / "s"), "AI_DATA": "x"}, tmp_path / "s"),
                          ({}, Path.home() / "ai-data" / "datasets")):
        monkeypatch.delenv("AI_DATA", raising=False)
        monkeypatch.delenv("DATASET_STORE", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert store_path() == Settings().dataset_store == expected
        assert DatasetCatalog("http://x", app="a").store == store_path()


def test_the_fake_can_be_preloaded_and_records_calls(tmp_path):
    fake = FakeDatasetCatalog(app="concord-bench")
    assert fake.store.is_dir()
    s = fake.add_audio(A)
    assert fake.audio(s)["duration"] == 0.5 and fake.audio_path(s).read_bytes() == A
    m = fake.add_audio(b"mp3 bytes", ".mp3", duration=2.0, sample_rate=44100)
    assert fake.audio(m) == {**fake.audio(m), "ext": ".mp3", "duration": 2.0, "sample_rate": 44100}
    [(_, a)] = files(tmp_path, ("a.wav", A))
    assert fake.upload_audio(a) == s and fake.calls == []  # in the store already: not uploaded
    fake.hold([s])
    assert fake.held() == {s} and fake.held("concord-label") == set()
    did = fake.create_dataset("Calls")["id"]
    fake.publish_audio(did, {"a.wav": s, "m.mp3": m})
    assert [c[0] for c in fake.calls] == ["hold", "create_dataset", "publish_audio"]
    root = fake.store
    del fake
    import gc

    gc.collect()
    assert not root.exists()


def test_the_fake_can_be_preloaded_with_versions():
    fake = FakeDatasetCatalog(app="concord-bench")
    did = fake.create_dataset("Calls", language="en")["id"]
    v1 = fake.add_version(did, {"a.wav": A, "b.wav": B}, ASR, language="en")
    assert (v1["n"], v1["label_types"], v1["counts"]["files"], v1["manifest"]["language"]) == (1, ["asr"], 2, "en")
    assert check_folder(fake.version_dir(did, 1)) == [] and fake.missing([sha(A), sha(B)]) == []
    v2 = fake.add_version(did, {"a.wav": A}, label_types=())
    assert (v2["n"], v2["parent_n"], v2["format_version"], v2["label_types"]) == (2, 1, 3, [])
    assert check_folder(fake.version_dir(did, 2)) == [] and fake.calls == [("create_dataset", ("Calls",))]
