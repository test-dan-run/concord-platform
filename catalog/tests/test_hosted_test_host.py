"""POST /api/hosted/test sends a stored key only to the host it was stored for."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}


@pytest.fixture
def seen(cat):
    sent = []

    def provider(r):
        r.read()
        sent.append((str(r.url), r.headers.get("authorization")))
        return httpx.Response(200, json={"data": [{"id": "whisper-1"}]})

    cat.hosted.transport = httpx.MockTransport(provider)
    return sent


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


@pytest.mark.parametrize("base_url", ["https://attacker.example/v1", "http://api.example.com/v1",
                                      "https://api.example.com:8443/v1", "https://api.example.com.attacker.example/v1"])
def test_the_stored_key_is_not_sent_to_another_host(api, seen, base_url):
    assert api.post("/api/hosted", json=ASR).status_code == 201
    seen.clear()
    r = api.post("/api/hosted/test", json={"id": "hosted-asr", "base_url": base_url})
    assert r.status_code == 422 and r.json()["error"]["message"] == "enter the key to test a different host"
    assert seen == []


def test_the_stored_key_is_used_on_the_same_host_and_a_new_key_anywhere(api, seen):
    assert api.post("/api/hosted", json=ASR).status_code == 201
    seen.clear()
    same = api.post("/api/hosted/test", json={"id": "hosted-asr", "base_url": "https://API.example.com:443/v2/"})
    assert same.status_code == 200 and seen[0] == ("https://api.example.com/v2/models", f"Bearer {KEY}")
    seen.clear()
    other = api.post("/api/hosted/test", json={"id": "hosted-asr", "base_url": "https://eu.example.com/v1", "key": "sk-eu-1"})
    assert other.status_code == 200 and seen[0] == ("https://eu.example.com/v1/models", "Bearer sk-eu-1")
