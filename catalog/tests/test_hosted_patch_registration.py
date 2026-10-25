"""PATCH /api/hosted/{id} edits display_name, tags and parent_id like PATCH /api/models/{id}."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "display_name": "H", "tags": ["asr"], "data_leaves_network": True}


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    with TestClient(create_app(cat, start=False)) as c:
        assert c.post("/api/hosted", json=ASR).status_code == 201
        yield c


def test_display_name_tags_and_parent_are_written(api):
    r = api.patch("/api/hosted/hosted-asr", json={"display_name": "Renamed", "tags": ["x", "x", "y"],
                                                   "parent_id": "default-asr"})
    assert r.status_code == 200
    assert (r.json()["display_name"], r.json()["tags"], r.json()["parent_id"]) == ("Renamed", ["x", "y"], "whisper-large-v3")
    r = api.patch("/api/hosted/hosted-asr", json={"parent_id": None, "max_concurrency": 3})  # null clears, the rest stays
    assert (r.json()["display_name"], r.json()["tags"], r.json()["parent_id"]) == ("Renamed", ["x", "y"], None)
    assert r.json()["hosted"]["max_concurrency"] == 3


def test_an_unknown_parent_is_a_404_that_saves_nothing(api, cat):
    r = api.patch("/api/hosted/hosted-asr", json={"parent_id": "no-such-model", "max_concurrency": 8, "tags": ["z"]})
    assert r.status_code == 404
    rec = api.get("/api/models/hosted-asr").json()
    assert rec["hosted"]["max_concurrency"] == 2 and rec["tags"] == ["asr"] and rec["parent_id"] is None
