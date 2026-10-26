"""A PATCH that points an OIP endpoint at another model needs that model's signature: when the provider's metadata
for it cannot be read, the PATCH is a 422 like create, and the old model's signature is never kept for it."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": "sk-test-0123456789abcdef7f3a", "timeout_s": 5,
       "max_concurrency": 2, "max_upload_mb": 1, "data_leaves_network": True}
OIP = {"contract": "oip-v2", "api": "infer", "base_url": "https://api.example.com", "model": "vad-x"}
SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]}],
       "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}]}


def provider(r: httpx.Request) -> httpx.Response:
    r.read()
    if r.url.path == "/v2/models/vad-x/ready":
        return httpx.Response(200, json={"ready": True})
    if r.url.path == "/v2/models/vad-x":
        return httpx.Response(200, json={"name": "vad-x", **SIG})
    return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(provider)
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_an_oip_model_change_without_readable_metadata_is_a_422(api, cat):
    assert api.post("/api/hosted", json={**ASR, **OIP}).json()["signature"] == SIG
    r = api.patch("/api/hosted/hosted-asr", json={"model": "no-such"})
    assert r.status_code == 422 and r.json()["error"]["message"].startswith("cannot read no-such's OIP metadata")
    h = cat.hosted.get("hosted-asr")
    assert h.model == "vad-x" and h.signature == SIG  # nothing was saved
    r = api.patch("/api/hosted/hosted-asr", json={"timeout_s": 9})  # the same build keeps its signature
    assert r.status_code == 200 and r.json()["signature"] == SIG
