"""Catalog fixtures: the real manifests, a temp AI_ASSETS with tiny weight folders, a fake runtime and
a scheduler driven by hand (no background threads), so every test is deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from catalog.app import build_catalog
from catalog.runtime import FakeRuntime
from catalog.settings import Settings

# Engine models (and custom "existing local copy" folders) used by the tests: a small file each.
LOCAL = ["whisper-large-v3", "qwen3-32b-awq", "qwen3-8b-fp8", "qwen2.5-14b-instruct-awq", "bge-m3",
         "pyannote-speaker-diarization-3.1"]


@pytest.fixture
def assets(tmp_path, monkeypatch) -> Path:
    a = tmp_path / "ai-assets"
    for d in LOCAL:
        (a / d).mkdir(parents=True)
        (a / d / "model.safetensors").write_bytes(d.encode() * 10)
        (a / d / "SOURCE.txt").write_text("repo: x\n")
    monkeypatch.setenv("AI_ASSETS", str(a))
    return a


@pytest.fixture
def master_key(tmp_path) -> Path:
    from cryptography.fernet import Fernet

    path = tmp_path / "secrets" / "catalog.key"
    path.parent.mkdir()
    path.write_bytes(Fernet.generate_key())
    return path


@pytest.fixture
def settings(tmp_path, assets, master_key) -> Settings:
    return Settings(db_path=tmp_path / "catalog.db", docker=False, reconcile_s=0.05, settle_min_s=0.0,
                    verify_on_new_build=False, ui_dir=tmp_path / "no-ui", public_url="http://catalog.test",
                    master_key_file=master_key, export_dir=tmp_path / "exports",
                    dataset_store=tmp_path / "datasets")


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def cat(settings, runtime):
    c = build_catalog(settings, runtime)
    yield c
    c.db.close()
