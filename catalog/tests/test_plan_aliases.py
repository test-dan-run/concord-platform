"""POST /api/jobs rejects a malformed body with 422. Aliases in plans: test_plan_residency.py."""

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("body", [
    {"stages": "abc"},
    {"stages": [["x"]]},
    {"stages": [{"name": "a", "models": "silero-vad"}]},
    {"stages": [{"name": "a", "models": [1]}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": "nope"}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": {"silero-vad": "nope"}}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": {"silero-vad": ["nope"]}}]},
])
def test_a_malformed_plan_body_is_rejected(api, body):
    r = api.post("/api/jobs", json=body)
    assert r.status_code == 422, (body, r.status_code, r.text)
