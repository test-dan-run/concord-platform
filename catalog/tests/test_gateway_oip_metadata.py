"""GET /v2/models/{id} (OIP model metadata) goes through the gateway like /v2/models/{id}/..."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

META = {"name": "vad-x", "inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]}],
        "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}]}
VAD = {"id": "remote-vad", "contract": "oip-v2", "api": "infer", "base_url": "https://inference.example.com",
       "model": "vad-x", "auth_type": "bearer", "key": "sk-test-0123456789abcdef7f3a", "data_leaves_network": True}


@pytest.fixture
def seen(cat):
    sent = []

    def upstream(r: httpx.Request) -> httpx.Response:
        r.read()
        sent.append(str(r.url))
        if r.url.path.endswith("/ready"):
            return httpx.Response(200, json={"ready": True})
        return httpx.Response(200, json=META)

    cat.hosted.transport = httpx.MockTransport(upstream)
    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    cat.scheduler.is_up = lambda name: True
    return sent


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), follow_redirects=False) as c:
        yield c


def test_platform_model_metadata(api, cat, seen):
    r = api.get("/v2/models/silero-vad")
    assert r.status_code == 200 and r.json() == META
    assert seen == [f"{cat.scheduler.upstream_url('silero-vad')}/v2/models/silero-vad"]


def test_hosted_model_metadata_names_the_provider_model(api, seen):
    assert api.post("/api/hosted", json=VAD).status_code == 201
    seen.clear()
    r = api.get("/v2/models/remote-vad")
    assert r.status_code == 200 and r.json() == META
    assert seen == ["https://inference.example.com/v2/models/vad-x"]
