"""Bundles are removed when CATALOG_EXPORT_DIR is a symlink or a relative path."""

from pathlib import Path

import pytest
from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)


@pytest.mark.parametrize("kind", ["symlink", "relative"])
def test_deleting_a_bundle_removes_its_folder(api, cat, images, tmp_path, monkeypatch, kind):
    real = tmp_path / "real-exports"
    real.mkdir()
    if kind == "symlink":
        cat.settings.export_dir = tmp_path / "exports-link"
        cat.settings.export_dir.symlink_to(real)
    else:
        monkeypatch.chdir(tmp_path)
        cat.settings.export_dir = Path("real-exports")
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])
    assert job["state"] == "done", job
    folder = real / job["name"]
    assert folder.is_dir()
    assert api.delete(f"/api/exports/{job['id']}").status_code == 204
    assert not folder.exists()


def test_a_folder_outside_the_export_dir_is_never_removed(cat, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cat.exporter._remove(str(outside))
    assert outside.is_dir()
