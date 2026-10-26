"""Bundles are removed when CATALOG_EXPORT_DIR is a symlink (both sides are resolved, which also covers a relative
path), and a folder outside it never is."""

from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)


def test_deleting_a_bundle_removes_its_folder(api, cat, images, tmp_path):
    real = tmp_path / "real-exports"
    real.mkdir()
    cat.settings.export_dir = tmp_path / "exports-link"
    cat.settings.export_dir.symlink_to(real)
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
