"""The importer never writes outside services/ and $AI_ASSETS for a bundle's ids and weights paths, and a
broken bundle gets a message, not a traceback."""

import json
import shutil

import pytest

from catalog import importer
from tools.schema import PLATFORM_DIR


@pytest.fixture
def target(tmp_path, monkeypatch):
    t = tmp_path / "target"
    t.mkdir()
    for part in ("config.yaml", "engines", "contracts", "custom/_bases", "custom/wpe-dereverb",
                 "custom/faster-whisper-large-v3"):
        src = PLATFORM_DIR / part
        (shutil.copytree if src.is_dir() else shutil.copy)(src, t / part)
    (t / "services").mkdir()
    shutil.copy(PLATFORM_DIR / "services" / "wpe-dereverb.yaml", t / "services")
    (tmp_path / "assets").mkdir()
    monkeypatch.setenv("AI_ASSETS", str(tmp_path / "assets"))
    return t


def make_bundle(root, models):
    """models: {id: service.yaml text or None}; each gets a weights/ folder with one file."""
    for mid, text in models.items():
        d = root / "models" / mid
        (d / "weights").mkdir(parents=True)
        (d / "weights" / "model.bin").write_bytes(b"w" * 10)
        if text is not None:
            (d / "service.yaml").write_text(text)
    (root / "manifest.json").write_text(json.dumps({"format": 1, "models": [{"id": m} for m in models]}))
    return root


@pytest.mark.parametrize("bad", ["../evil", "Evil", "a/b", "", None])
def test_a_model_id_that_is_not_kebab_case_is_refused(tmp_path, target, bad):
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "evil").mkdir()
    (bundle / "evil" / "service.yaml").write_text("anything: the bundle wants\n")
    (bundle / "manifest.json").write_text(json.dumps({"format": 1, "models": [{"id": bad}]}))
    with pytest.raises(importer.BundleError, match="kebab-case"):
        importer.import_bundle(bundle, target, catalog=None, db_path=tmp_path / "t.db", images=False)
    assert sorted(p.name for p in target.iterdir()) == ["config.yaml", "contracts", "custom", "engines", "services"]
    assert [p.name for p in (target / "services").iterdir()] == ["wpe-dereverb.yaml"]


def test_weights_outside_ai_assets_are_refused(tmp_path, target):
    text = (PLATFORM_DIR / "services" / "faster-whisper-large-v3.yaml").read_text()
    text = text.replace("path: ${AI_ASSETS}/faster-whisper-large-v3", "path: ${AI_ASSETS}/../outside")
    bundle = make_bundle(tmp_path / "bundle", {"faster-whisper-large-v3": text})
    with pytest.raises(importer.BundleError, match="outside AI_ASSETS"):
        importer.import_bundle(bundle, target, catalog=None, db_path=tmp_path / "t.db", images=False)
    assert not (tmp_path / "outside").exists() and not (tmp_path / "t.db").exists()


def test_weights_under_ai_assets_are_copied(tmp_path, target):
    text = (PLATFORM_DIR / "services" / "faster-whisper-large-v3.yaml").read_text()
    bundle = make_bundle(tmp_path / "bundle", {"faster-whisper-large-v3": text})
    assert importer.import_bundle(bundle, target, catalog=None, db_path=tmp_path / "t.db",
                                  images=False) == ["faster-whisper-large-v3"]
    assert (tmp_path / "assets" / "faster-whisper-large-v3" / "model.bin").read_bytes() == b"w" * 10


@pytest.mark.parametrize("make", ["not a tar", "bad json", "no models"])
def test_a_broken_bundle_is_a_message_not_a_traceback(tmp_path, target, capsys, make):
    if make == "not a tar":
        path = tmp_path / "x.tar"
        path.write_bytes(b"this is not a tar file")
    else:
        path = tmp_path / "bundle"
        path.mkdir()
        (path / "manifest.json").write_text("{not json" if make == "bad json" else json.dumps({"format": 1}))
    assert importer.main([str(path), "--platform-dir", str(target), "--no-images", "--catalog", ""]) == 1
    assert capsys.readouterr().err.startswith("import failed: ")
