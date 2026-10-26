"""A base_url the catalog cannot read as a URL with a host is refused with a 422, so it is never stored and the model
list keeps working."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": "sk-test-0123456789abcdef7f3a", "timeout_s": 5,
       "max_concurrency": 2, "max_upload_mb": 1, "data_leaves_network": True}
BAD = ["https://[api.example.com/v1", "https://api.example.com:99999/v1", "https://api.example.com:abc/v1",
       "https:///v1", "https://:443/v1"]


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("url", BAD)
def test_a_base_url_without_a_readable_host_is_refused(api, url):
    body = {**ASR, "base_url": url}
    for r in (api.post("/api/hosted/test", json=body), api.post("/api/hosted", json=body)):
        assert r.status_code == 422 and r.json()["error"]["message"].startswith("base_url "), r.text
    assert api.get("/api/models").status_code == 200 and api.get("/api/models/hosted-asr").status_code == 404
    assert api.post("/api/hosted", json=ASR).status_code == 201
    r = api.patch("/api/hosted/hosted-asr", json={"base_url": url})
    assert r.status_code == 422 and r.json()["error"]["message"].startswith("base_url ")
    assert api.get("/api/models").status_code == 200
    assert api.get("/api/models/hosted-asr").json()["hosted"]["base_url"] == ASR["base_url"]
