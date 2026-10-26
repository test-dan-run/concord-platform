"""A sha256 is exactly 64 lowercase hex characters: one with a newline after it is refused by the catalog's store
routes and by FakeDatasetCatalog alike (files.csv's check is in test_dataset_fake_rules)."""

import pytest
from catalog_helpers import sha, upload, wav
from fastapi.testclient import TestClient

from aicore.datasets import DatasetCatalogError, FakeDatasetCatalog

from catalog.app import create_app

A = wav(1)
NEWLINE = "a" * 64 + "\n"


def test_the_store_routes_refuse_a_sha256_with_a_newline(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as api:
        upload(api, A)
        r = api.post("/api/store/holds", json={"app": "a", "add": [NEWLINE]})
        assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request"
        assert api.post("/api/store/holds", json={"app": "a", "remove": [sha(A) + "\n"]}).status_code == 422
        assert api.post("/api/store/missing", json={"sha256": [NEWLINE]}).status_code == 422
        assert api.get("/api/store/audio/" + sha(A) + "%0A").status_code == 422
        assert api.post("/api/store/holds", json={"app": "a", "add": [sha(A)]}).json() == {"held": 1}
    app.state.datasets.db.close()


def test_holds_and_missing_refuse_a_sha256_with_a_newline(tmp_path):
    dc = FakeDatasetCatalog(tmp_path / "fake-store", app="concord-label")
    for call in (dc.hold, dc.release, dc.missing):
        with pytest.raises(DatasetCatalogError) as exc:
            call([NEWLINE])
        assert exc.value.status == 422
    assert dc.hold([]) == 0

