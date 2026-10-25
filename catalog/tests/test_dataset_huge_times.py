"""A references time written as an integer too large for a float is a problem, not a 500: float() raises
OverflowError for it, which is not a ValueError."""

import pytest
from catalog_helpers import new_dataset, publish, upload, version_zip, wav
from fastapi.testclient import TestClient

from catalog.app import create_app

A = wav(1)
HUGE = "1" + "0" * 400


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        upload(c, A)
        yield c
    app.state.datasets.db.close()


def test_a_time_too_large_for_a_float_is_422(api):
    refs = {"segments.jsonl": '{"file": "a.wav", "start": 0.1, "end": ' + HUGE + ', "text": "hello"}\n'}
    did = new_dataset(api)
    r = publish(api, did, version_zip([("a.wav", A)], refs))
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version", r.text
    assert r.json()["error"]["problems"][0].startswith("refs/segments.jsonl: can't be read")
    assert api.get(f"/api/datasets/{did}").json()["versions"] == []
