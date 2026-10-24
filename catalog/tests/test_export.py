"""Export jobs with a FakeRuntime: shared image layers written once, job states, cancel, restart
recovery, the disk space check, the registry destination and the streamed download."""

import io
import json
import tarfile
import time
from collections import namedtuple

import pytest
from catalog_helpers import wait_builds
from fastapi.testclient import TestClient

from catalog import exporter as exporter_mod
from catalog.app import create_app

PAIR = ["silero-vad", "pyannote-diarization"]


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


@pytest.fixture
def images(cat, runtime):
    """The two models' images share a 5000-byte base layer; pyannote's weights.yaml files are fetched."""
    import yaml

    from tools import identity

    p = cat.scheduler.platform()
    s = p.get("pyannote-diarization")
    for f in yaml.safe_load(identity.weights_source(p, s).path.read_text())["files"]:
        target = p.weights_dir(s) / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w" * 100)
    a, b = (p.image(p.get(m)) for m in PAIR)
    runtime.layers[a] = [("base", 5000), ("vad", 1000)]
    runtime.layers[b] = [("base", 5000), ("diar", 2000)]
    wait_builds(cat, PAIR)
    return a, b


def finish(cat, api, job_id, timeout=10.0):
    cat.exporter.wait(timeout)
    return api.get(f"/api/exports/{job_id}").json()


def test_bundle_writes_shared_layers_once_and_streams_one_tar(api, cat, images, settings):
    listed = {m["id"]: m for m in api.get("/api/export/models").json()}
    assert listed["silero-vad"]["image_size"] == 6000 and listed["pyannote-diarization"]["image_size"] == 7000
    assert listed["silero-vad"]["exportable"] and listed["pyannote-diarization"]["weights_size"] > 0
    api.post("/api/models/silero-vad/register", json={"tags": ["vad"], "display_name": "Silero"})
    r = api.post("/api/exports", json={"models": PAIR})
    assert r.status_code == 202 and r.json()["state"] in ("queued", "running", "done")
    job = finish(cat, api, r.json()["id"])
    assert job["state"] == "done" and job["available"], job
    root = settings.export_dir / job["name"]
    assert (root / "images.tar").stat().st_size == 8000  # base once: 5000 + 1000 + 2000, not 13000
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["images"] == "images.tar" and [m["id"] for m in manifest["models"]] == PAIR
    assert all(m["build_key"] and m["image_id"] for m in manifest["models"])
    reg = json.loads((root / "models" / "silero-vad" / "registration.json").read_text())
    assert reg["display_name"] == "Silero" and reg["tags"] == ["vad"]
    assert listed["silero-vad"]["weights_note"] == "downloaded on first start"  # its weights.yaml files are not fetched
    assert any((root / "models" / "pyannote-diarization" / "weights").rglob("*"))
    assert not (root / "models" / "silero-vad" / "weights").exists()
    assert (root / "models" / "silero-vad" / "service.yaml").read_text().startswith("name: silero-vad")
    d = api.get(f"/api/exports/{job['id']}/download")
    assert d.status_code == 200 and d.headers["content-disposition"].endswith(f'{job["name"]}.tar"')
    names = tarfile.open(fileobj=io.BytesIO(d.content)).getnames()
    assert f"{job['name']}/images.tar" in names and f"{job['name']}/models/silero-vad/service.yaml" in names
    assert [j["id"] for j in api.get("/api/exports").json()] == [job["id"]]
    assert api.delete(f"/api/exports/{job['id']}").status_code == 204 and not root.exists()


def test_one_export_at_a_time_and_cancel_removes_the_partial_bundle(api, cat, images, runtime, settings):
    runtime.save_delay_s = 5.0
    job = api.post("/api/exports", json={"models": PAIR}).json()
    deadline = time.monotonic() + 5
    while api.get(f"/api/exports/{job['id']}").json()["phase"] != "saving images (one archive, shared layers once)":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert api.post("/api/exports", json={"models": PAIR}).status_code == 409
    assert api.delete(f"/api/exports/{job['id']}").status_code == 204  # running: cancels
    done = finish(cat, api, job["id"])
    assert done["state"] == "cancelled" and done["failed_phase"] == "saving images"
    assert not (settings.export_dir / done["name"]).exists()
    assert api.get(f"/api/exports/{job['id']}/download").status_code == 410


def test_disk_space_is_checked_first_and_the_failure_names_the_phase(api, cat, images, monkeypatch, settings):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(exporter_mod.shutil, "disk_usage", lambda p: Usage(10_000, 9_000, 1_000))
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])
    assert job["state"] == "failed" and job["failed_phase"] == "checking disk space"
    assert "free" in job["error"] and "needed" in job["error"] and not (settings.export_dir / job["name"]).exists()


def test_a_restart_fails_the_running_job(cat, settings):
    partial = settings.export_dir / "catalog-export-2026-10-23"
    partial.mkdir(parents=True)
    cat.db.execute("INSERT INTO export_job (id, request, state, phase, path, created_at) VALUES "
                   "('x-1', '{\"models\": []}', 'running', 'copying weights for silero-vad', ?, '2026-10-23T00:00:00Z')",
                   (str(partial),))
    cat.exporter.recover()
    job = cat.exporter.get("x-1")
    assert job["state"] == "failed" and job["error"] == "interrupted by a catalog restart" and not partial.exists()


def test_registry_destination_pushes_images_and_bundles_the_rest(api, cat, images, runtime, settings):
    assert api.post("/api/exports", json={"models": PAIR, "destination": "registry"}).status_code == 422
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR, "destination": "registry",
                                                          "registry": "registry.example.internal/models/",
                                                          "weights": False}).json()["id"])
    assert job["state"] == "done", job
    assert [t for _, t in runtime.pushed] == [f"registry.example.internal/models/{ref.rsplit('/', 1)[-1]}" for ref in images]
    root = settings.export_dir / job["name"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["images"] is None and not (root / "images.tar").exists()
    assert manifest["models"][0]["pushed"]["ref"].startswith("registry.example.internal/models/")
    assert not (root / "models" / "pyannote-diarization" / "weights").exists()


def test_models_without_a_local_image_or_hosted_cannot_be_exported(api, cat, images, runtime):
    runtime.missing.add(images[0])
    cat.exporter._sizes.clear()
    m = {x["id"]: x for x in api.get("/api/export/models").json()}["silero-vad"]
    assert not m["exportable"] and "not present locally" in m["reason"]
    assert api.post("/api/exports", json={"models": ["silero-vad"]}).status_code == 422


def test_import_loads_a_bundle_into_another_platform(api, cat, images, settings, tmp_path, monkeypatch):
    import shutil
    import subprocess

    from catalog import importer
    from catalog.db import DB
    from tools.schema import PLATFORM_DIR, load_platform

    api.post("/api/models/pyannote-diarization/register", json={"tags": ["diarisation"], "display_name": "Pyannote"})
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])
    tar = tmp_path / "carried" / f"{job['name']}.tar"
    tar.parent.mkdir()
    tar.write_bytes(api.get(f"/api/exports/{job['id']}/download").content)

    target = tmp_path / "target"  # another host: the same platform code, none of these services yet
    target.mkdir()
    for part in ("config.yaml", "engines", "contracts", "custom/_bases", "custom/silero-vad", "custom/pyannote-diarization"):
        src = PLATFORM_DIR / part
        (shutil.copytree if src.is_dir() else shutil.copy)(src, target / part)
    (target / "services").mkdir()
    shutil.copy(PLATFORM_DIR / "services" / "wpe-dereverb.yaml", target / "services")
    shutil.copytree(PLATFORM_DIR / "custom" / "wpe-dereverb", target / "custom" / "wpe-dereverb")
    assets = tmp_path / "target-assets"
    monkeypatch.setenv("AI_ASSETS", str(assets))
    with pytest.raises(importer.BundleError, match="does not exist"):  # never create a stray weights tree
        importer.import_bundle(tar, target, catalog=None, db_path=tmp_path / "target.db",
                               run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    assert [f.name for f in (target / "services").glob("*.yaml")] == ["wpe-dereverb.yaml"]  # nothing written
    docker = []

    def run(cmd, **kw):
        docker.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assets.mkdir()
    ids = importer.import_bundle(tar, target, catalog=None, db_path=tmp_path / "target.db", run=run)
    assert ids == PAIR and docker[0][:3] == ["docker", "load", "-i"] and docker[0][3].endswith("/images.tar")
    p = load_platform(target)
    assert {"silero-vad", "pyannote-diarization", "wpe-dereverb"} <= {s.name for s in p.services}
    assert any(p.weights_dir(p.get("pyannote-diarization")).rglob("*"))
    db = DB(tmp_path / "target.db")
    row = db.one("SELECT * FROM registration WHERE model_id = 'pyannote-diarization'")
    assert row["display_name"] == "Pyannote" and json.loads(row["tags"]) == ["diarisation"]
    db.close()
    # a second import finds everything in place; a conflicting manifest is kept unless --force
    (target / "services" / "silero-vad.yaml").write_text((target / "services" / "silero-vad.yaml").read_text() + "\n# local edit\n")
    assert importer.import_bundle(tar, target, catalog=None, db_path=tmp_path / "target.db", run=run) == ["pyannote-diarization"]


def test_import_reads_platform_env_and_refuses_a_missing_assets_folder(tmp_path, monkeypatch):
    from catalog import importer

    (tmp_path / ".env").write_text("# comment\nAI_ASSETS=/somewhere/assets\nexport AI_HOST='localhost'\nEMPTY=\n")
    monkeypatch.delenv("AI_ASSETS", raising=False)
    monkeypatch.setenv("AI_HOST", "already-set")
    importer.load_dotenv(tmp_path)
    import os

    assert os.environ["AI_ASSETS"] == "/somewhere/assets" and os.environ["AI_HOST"] == "already-set"
    assert "EMPTY" not in os.environ
