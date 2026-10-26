"""FakeDatasetCatalog keeps the catalog service's version rules: both check a folder with the same check_folder and
read references with the same parse (aicore.dataset_format), so an app test can't pass with a folder the service
refuses. A refused publish leaves nothing behind, and the errors carry the service's messages."""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from catalog_helpers import ASR, jsonl, sha, wav
from fastapi.testclient import TestClient

from aicore.datasets import DatasetCatalog, DatasetCatalogError, FakeDatasetCatalog, InvalidVersion, StaleParent
from aicore.datasets import digest as client_digest

from catalog.app import create_app
from catalog.datasets.versions import check_folder
from catalog.datasets.versions import digest as server_digest

A, B, C = wav(1), wav(2), wav(3)
FILES = [("a.wav", A), ("b.wav", B)]
SOURCE = {"app": "x", "install_id": "y", "dataset_uid": "z", "dataset_name": "w"}


@contextlib.contextmanager
def _server(cat) -> Iterator[DatasetCatalog]:
    app = create_app(cat, start=False)
    with TestClient(app) as http:
        yield DatasetCatalog("http://testserver", app="concord-label", store=cat.settings.dataset_store, http=http)
    app.state.datasets.db.close()


@pytest.fixture(params=["fake", "real"])
def dc(request, tmp_path):
    if request.param == "fake":
        yield FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")
    else:
        with _server(request.getfixturevalue("cat")) as c:
            yield c


def folder(tmp_path: Path, refs: dict | None = None, *, items=FILES, label_types=("asr",), format_version=2,
           audio=False, files_csv: bytes | None = None, **manifest) -> Path:
    """A version folder as an app writes one; `refs` values may be bytes and names may hold a folder."""
    root = tmp_path / "out" / f"v{len(list((tmp_path / 'out').glob('*'))) if (tmp_path / 'out').exists() else 0}"
    (root / "refs").mkdir(parents=True)
    m = {"format": "concord-reference-package", "format_version": format_version, "name": "Calls",
         "created_at": "2026-10-24T09:00:00Z", "label_types": list(label_types),
         "counts": {"files": len(items), "segments": 0, "hours": 0.0},
         "audio_included": "all" if audio else "none", **manifest}
    (root / "manifest.json").write_text(json.dumps(m))
    (root / "files.csv").write_bytes(files_csv or ("path,sha256,duration,sample_rate\n" +
                                                   "".join(f"{p},{sha(d)},0.5,16000\n" for p, d in items)).encode())
    for name, text in (refs or {}).items():
        (root / "refs" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "refs" / name).write_bytes(text if isinstance(text, bytes) else text.encode())
    for p, d in items if audio else []:
        (root / "audio" / p).parent.mkdir(parents=True, exist_ok=True)
        (root / "audio" / p).write_bytes(d)
    return root


def _dataset(dc, tmp_path: Path, *items: tuple[str, bytes]) -> str:
    for name, data in items or FILES:
        (tmp_path / name).write_bytes(data)
        dc.upload_audio(tmp_path / name)
    return dc.create_dataset("Calls")["id"]


# Folders the service refuses and the fake took (or crashed on) before it shared the service's check. The service
# uses the same check_folder, so the grid runs on the fake and one case runs on the service.
REFUSED = {
    "scope 'partial'": dict(refs=ASR, scope="partial"),
    "source is a string": dict(refs=ASR, source="label", export_version=1),
    "source without export_version": dict(refs=ASR, source=SOURCE),
    "source without install_id": dict(refs=ASR, source={"app": "x"}, export_version=1),
    "counts is a list": dict(refs=ASR, counts=[2]),
    "audio_included 'partial' with all the audio": dict(refs=ASR, audio=True, audio_included="partial"),
    "refs/ holds a folder": dict(refs={**ASR, "sub/x.txt": "x"}),
    "attributes.json names an unknown path": dict(refs={**ASR, "attributes.json": json.dumps(
        {"files": {"zzz.wav": {"a": 1}}})}),
    "attributes.json has a null label": dict(refs={**ASR, "attributes.json": json.dumps(
        {"files": {"a.wav": {"note": None}}})}),
    "attributes.json is a list": dict(refs={**ASR, "attributes.json": "[]"}),
    "attributes.json is not JSON": dict(refs={**ASR, "attributes.json": "{nope"}),
    "a uids line without uid": dict(refs={**ASR, "uids.jsonl": jsonl([{"file": "a.wav", "start": 0, "end": 1}])}),
    "a segments record without start": dict(refs={"segments.jsonl": jsonl([{"file": "a.wav", "text": "hi"}])}),
    "an RTTM start that is not a number": dict(refs={"refs.rttm": "SPEAKER a 1 x 1 <NA> <NA> s1 <NA> <NA>\n"},
                                               label_types=("diar",)),
    "a sha256 with a newline": dict(
        refs=ASR, items=[("a.wav", A)], counts={"files": 1},
        files_csv=f'path,sha256,duration,sample_rate\na.wav,"{sha(A)}\n",0.5,16000\n'.encode()),
}


def _refused(dc, tmp_path: Path, case: str) -> None:
    did = _dataset(dc, tmp_path)
    root = folder(tmp_path, **REFUSED[case])
    with pytest.raises(InvalidVersion) as exc:
        dc.publish(did, root)
    assert exc.value.status == 422 and exc.value.problems == check_folder(root, package=True, hash_audio=False)
    assert exc.value.message == "the version is invalid: " + "; ".join(exc.value.problems[:10])
    assert dc.dataset(did)["versions"] == [] and not (dc.store / "versions" / did / "1").exists()


@pytest.mark.parametrize("case", REFUSED)
def test_a_folder_the_service_refuses_is_refused_by_the_fake_with_the_same_problems(tmp_path, case):
    _refused(FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label"), tmp_path, case)


def test_the_service_refuses_a_folder_with_the_same_problems(cat, tmp_path):
    with _server(cat) as server:
        _refused(server, tmp_path, "attributes.json is not JSON")


def test_a_bad_references_file_leaves_the_dataset_usable(dc, tmp_path):
    did = _dataset(dc, tmp_path)
    with pytest.raises(InvalidVersion):
        dc.publish(did, folder(tmp_path, {"segments.jsonl": "{nope\n"}))
    with pytest.raises(InvalidVersion):
        dc.publish(did, folder(tmp_path, {**ASR, "attributes.json": "[]"}))
    assert not dc.version_dir(did, 1).exists()
    v = dc.publish(did, folder(tmp_path, ASR))
    assert (v["n"], v["created"], v["labels"]) == (1, True, ["noise"])
    assert check_folder(dc.version_dir(did, 1)) == []
    assert [p.name for p in (dc.store / "tmp").iterdir()] == []


def test_errors_carry_the_services_messages(dc, tmp_path):
    did = _dataset(dc, tmp_path)
    bad = folder(tmp_path, ASR, items=[("a.wav", A)], audio=True, counts={"files": 1})
    (bad / "audio" / "a.wav").write_bytes(C)
    with pytest.raises(DatasetCatalogError) as exc:
        dc.publish(did, bad)
    assert (exc.value.status, exc.value.type) == (422, "hash_mismatch")
    assert exc.value.message == f"a.wav: the bytes hash to {sha(C)}, not {sha(A)}"
    (tmp_path / "c.wav").write_bytes(C)
    with pytest.raises(DatasetCatalogError) as exc:
        dc.upload_audio(tmp_path / "c.wav", "b" * 64)
    assert exc.value.message == f"the uploaded bytes hash to {sha(C)}, not {'b' * 64}; the upload was removed"
    dc.publish(did, folder(tmp_path, ASR))
    dc.publish(did, folder(tmp_path, ASR, description="v2"), parent=1)
    dc.delete_version(did, 2)
    with pytest.raises(StaleParent) as exc:
        dc.publish(did, folder(tmp_path, ASR, description="v3"), parent=2)
    assert exc.value.message == "v2 was deleted, but the latest version is v1; publish with force to keep it"


VALID = {
    "STM": dict(refs={"x.stm": ";; note\na 1 spk 0 0.4 <o,f0,male> hello there\nb 1 s2 0.1 0.2\n"}),
    "a format the catalog doesn't read": dict(refs={"refs.ctm": "a 1 0 1 hi\n"}),
}


@pytest.mark.parametrize("case", VALID)
def test_the_fake_reads_references_as_the_service_does(server_and_fake, tmp_path, case):
    out = []
    for dc in server_and_fake:
        did = _dataset(dc, tmp_path)
        v = dc.publish(did, folder(tmp_path, **VALID[case]))
        out.append((v["labels"], v["digest"], dc.files(did, 1)))
    assert out[0] == out[1]


@pytest.fixture
def server_and_fake(cat, tmp_path):
    with _server(cat) as server:
        yield server, FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")


# ----------------------------------------------------------------------------- the digest never changes

SAMPLE_DIGESTS = {
    "version-asr": "27178d53d6a7223430dff6f4bf23b500042fca48d22fe3666c19a67b9509c1e1",
    "version-audio-only": "17ee9bb72f737baf8a0feca7647d77e4d0ae4d581d15b69fddda0536b9cdfada",
    "version-diar-v1": "b7f4d12d8e26ba6f36913ebe5de143b3dfe455e3c660280d6aa103c05d07be61",
    "version-labelled": "1990589ed1175a9979c1b35532aa9ec67466343dd863173186368706ab3f0f98",
}


def test_the_digest_of_a_version_never_changes(tmp_path):
    """A stored version is found again by its digest, so these values are fixed: the samples, and a version with
    non-ASCII text, whose manifest is hashed with that text escaped (contracts/datasets/version.md §8)."""
    samples = Path(__file__).resolve().parents[2] / "contracts" / "datasets" / "samples"
    for impl in (server_digest, client_digest):
        assert {d.name: impl(d) for d in sorted(samples.iterdir()) if d.is_dir()} == SAMPLE_DIGESTS
    root = tmp_path / "v"
    (root / "refs").mkdir(parents=True)
    m = {"format": "concord-reference-package", "format_version": 2, "name": "Café 数据",
         "created_at": "2026-01-01T00:00:00Z", "label_types": ["asr"],
         "counts": {"files": 1, "segments": 1, "hours": 0.1}, "audio_included": "none", "tags": ["ü"],
         "source": {"app": "x"}, "export_version": 2}
    (root / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    (root / "files.csv").write_text("path,sha256,duration,sample_rate\ncafé.wav," + "a" * 64 + ",1.0,16000\n",
                                    encoding="utf-8")
    (root / "refs" / "segments.jsonl").write_text('{"file": "café.wav", "start": 0, "end": 1, "text": "é"}\n',
                                                  encoding="utf-8")
    for impl in (server_digest, client_digest):
        assert impl(root) == "2e5ab6513a5ea89859a04a30618f1258af15e089ac062aa8baa15d01f29294dc"
