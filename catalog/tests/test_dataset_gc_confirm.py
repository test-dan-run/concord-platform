"""Clean-up deletes only what the person agreed to: with expect_files, a real run that finds more files than the dry
run showed answers 409 gc_changed and deletes nothing."""

import pytest
from catalog_helpers import ASR, age, new_dataset, publish, sha, upload, version_zip, wav
from fastapi.testclient import TestClient

from catalog.app import create_app

CHANGED = "clean-up found more files than the check showed: check again"


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


def test_a_real_run_that_finds_more_than_expected_deletes_nothing(api):
    did = new_dataset(api)
    stray, listed = wav(40), wav(41)
    for d in (stray, listed):
        upload(api, d)
        age(api, sha(d))
    assert publish(api, did, version_zip([("a.wav", listed)], ASR)).status_code == 201
    dry = api.post("/api/store/gc", json={"dry_run": True}).json()
    assert dry["files"] == 1
    # The version goes (another tab, or an app) before the person presses "Delete 1 file".
    assert api.delete(f"/api/datasets/{did}/versions/1").status_code == 204
    r = api.post("/api/store/gc", json={"dry_run": False, "expect_files": 1})
    assert r.status_code == 409
    assert r.json() == {"error": {"message": CHANGED, "type": "gc_changed", "code": 409}}
    assert api.post("/api/store/missing", json={"sha256": [sha(stray), sha(listed)]}).json() == {"missing": []}
    # The new count deletes both; fewer than expected is fine too, and so is no count at all.
    assert api.post("/api/store/gc", json={"dry_run": False, "expect_files": 2}).json()["files"] == 2
    old = wav(42)
    upload(api, old)
    age(api, sha(old))
    assert api.post("/api/store/gc", json={"dry_run": False, "expect_files": 5}).json()["files"] == 1
    assert api.post("/api/store/gc", json={"dry_run": False, "expect_files": 0}).json()["files"] == 0
    upload(api, old)
    age(api, sha(old))
    assert api.post("/api/store/gc", json={"dry_run": False}).json()["files"] == 1
    # A dry run ignores expect_files, and a negative count is refused.
    later = wav(43)
    upload(api, later)
    age(api, sha(later))
    assert api.post("/api/store/gc", json={"dry_run": True, "expect_files": 0}).json()["files"] == 1
    assert api.post("/api/store/gc", json={"dry_run": False, "expect_files": -1}).status_code == 422
    assert api.post("/api/store/missing", json={"sha256": [sha(later)]}).json() == {"missing": []}
