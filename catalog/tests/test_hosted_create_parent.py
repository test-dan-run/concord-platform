"""Creating a hosted endpoint with an unknown parent_id is refused before anything is stored."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_an_unknown_parent_is_a_404_that_stores_nothing(api, cat):
    r = api.post("/api/hosted", json={**ASR, "parent_id": "no-such-model"})
    assert r.status_code == 404
    assert "hosted-asr" not in cat.hosted and cat.db.one("SELECT 1 FROM hosted") is None
    assert cat.db.one("SELECT 1 FROM registration") is None and cat.verifier.status("hosted-asr") is None
    assert api.get("/api/models/hosted-asr").status_code == 404
    again = api.post("/api/hosted", json={**ASR, "parent_id": "default-asr"})  # a retry with a known parent
    assert again.status_code == 201 and again.json()["parent_id"] == "whisper-large-v3"
