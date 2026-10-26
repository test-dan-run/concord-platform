"""An endpoint stored with a base_url the catalog now refuses (saved before base_url was checked): its connection test
answers 422, like a create would, instead of a 500, and it still lists, its host falling back to the URL as stored."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

GOOD = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
        "model": "whisper-1", "auth_type": "bearer", "key": "sk-test-0123456789abcdef7f3a", "data_leaves_network": True}


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        assert c.post("/api/hosted", json=GOOD).status_code == 201
        # saved before base_url was checked
        cat.db.execute("UPDATE hosted SET base_url = ? WHERE id = ?", ("https://[api.example.com/v1", "hosted-asr"))
        yield c


def test_the_connection_test_of_a_stored_bad_url_is_422(api):
    r = api.post("/api/models/hosted-asr/test")
    assert r.status_code == 422 and "base_url is not a valid URL" in r.text, r.text
    listed = api.get("/api/models")
    assert listed.status_code == 200
    assert listed.json()[0]["hosted"]["host"] == "https://[api.example.com/v1"  # unparseable: the URL as stored
