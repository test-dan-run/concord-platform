"""PATCH /api/hosted/{id} reads a new OIP signature when the endpoint becomes OIP, and drops it when it stops being OIP."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}
OIP = {"contract": "oip-v2", "api": "infer", "base_url": "https://api.example.com", "model": "vad-x"}
SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]}],
       "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}]}


def provider(r: httpx.Request) -> httpx.Response:
    r.read()
    if r.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "whisper-1"}]})
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


def test_a_patch_away_from_oip_drops_the_signature(api):
    assert api.post("/api/hosted", json={**ASR, **OIP}).json()["signature"] == SIG
    r = api.patch("/api/hosted/hosted-asr", json={k: ASR[k] for k in ("contract", "api", "base_url", "model")})
    assert r.status_code == 200 and r.json()["contract"] == "openai" and r.json()["signature"] is None
    assert api.get("/api/models/hosted-asr").json()["signature"] is None


def test_a_patch_to_oip_reads_the_signature(api):
    assert api.post("/api/hosted", json=ASR).status_code == 201
    r = api.patch("/api/hosted/hosted-asr", json=OIP)
    assert r.status_code == 200 and r.json()["contract"] == "oip-v2" and r.json()["signature"] == SIG
