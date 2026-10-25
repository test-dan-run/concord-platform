"""Export and import carry a weights folder's SOURCE.txt, so the model card keeps its provenance."""

import shutil
import subprocess

from catalog_helpers import wait_builds
from export_helpers import api, finish  # noqa: F401  (fixtures)

from tools import identity, model_cards
from tools.schema import PLATFORM_DIR, load_platform


def test_source_txt_goes_through_export_and_import(api, cat, runtime, settings, assets, tmp_path, monkeypatch):
    from catalog import importer

    p = cat.scheduler.platform()
    runtime.layers[p.image(p.get("whisper-large-v3"))] = [("vllm", 1000)]
    wait_builds(cat, ["whisper-large-v3"])
    (assets / "whisper-large-v3" / "SOURCE.txt").write_text("repo: openai/whisper-large-v3\nrevision: 06f233f\n")
    before = identity.content_hash(assets / "whisper-large-v3")
    job = finish(cat, api, api.post("/api/exports", json={"models": ["whisper-large-v3"]}).json()["id"])
    assert job["state"] == "done", job
    bundle = settings.export_dir / job["name"]
    weights = bundle / "models" / "whisper-large-v3" / "weights"
    assert sorted(f.name for f in weights.iterdir()) == ["SOURCE.txt", "model.safetensors"]

    target = tmp_path / "target"
    target.mkdir()
    for part in ("config.yaml", "engines", "contracts", "custom/_bases", "custom/wpe-dereverb"):
        src = PLATFORM_DIR / part
        (shutil.copytree if src.is_dir() else shutil.copy)(src, target / part)
    (target / "services").mkdir()
    shutil.copy(PLATFORM_DIR / "services" / "wpe-dereverb.yaml", target / "services")  # a platform must have one
    target_assets = tmp_path / "target-assets"
    target_assets.mkdir()
    monkeypatch.setenv("AI_ASSETS", str(target_assets))
    ids = importer.import_bundle(bundle, target, catalog=None, db_path=tmp_path / "target.db",
                                 run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    assert ids == ["whisper-large-v3"]
    copied = target_assets / "whisper-large-v3"
    assert (copied / "SOURCE.txt").read_text() == "repo: openai/whisper-large-v3\nrevision: 06f233f\n"
    tp = load_platform(target)
    src = model_cards.source_of(tp, tp.get("whisper-large-v3"))
    assert (src.origin, src.repo, src.ref) == ("SOURCE.txt", "openai/whisper-large-v3", "06f233f")
    assert identity.content_hash(copied) == before  # SOURCE.txt stays out of the weights hash
