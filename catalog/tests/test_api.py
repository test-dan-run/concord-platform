"""HTTP API, output index, planner and gateway, through FastAPI's TestClient."""

import httpx
import pytest
from catalog_helpers import step, wait_builds
from fastapi.testclient import TestClient

from aicore.catalog import Catalog, index_key

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_models_registration_and_filters(api, cat):
    assert api.get("/api/models").json() == []
    everything = api.get("/api/models", params={"all": 1}).json()
    assert {"whisper-large-v3", "silero-vad", "qwen3-8b"} <= {m["id"] for m in everything}
    r = api.post("/api/models/default-asr/register", json={"tags": ["call-centre"], "display_name": "Whisper"})
    assert r.status_code == 201
    rec = r.json()
    assert rec["id"] == "whisper-large-v3" and rec["aliases"] == ["default-asr"] and rec["registered"]
    assert rec["capabilities"]["status"] == "pending" and rec["gpu"] == {"fraction": 0.15, "gb": 4.8}
    assert rec["endpoint"] == {"gateway_url": "http://catalog.test/m/whisper-large-v3", "base_path": "/v1"}
    assert rec["cold_start_measured"] is False and rec["cold_start_s"] == 180.0
    assert [m["id"] for m in api.get("/api/models", params={"tag": "call-centre"}).json()] == ["whisper-large-v3"]
    assert api.get("/api/models", params={"task": "chat"}).json() == []
    assert api.patch("/api/models/whisper-large-v3", json={"tags": ["a", "a", "b"]}).json()["tags"] == ["a", "b"]
    assert api.get("/api/models/nope").status_code == 404
    assert api.get("/api/models/nope").json()["error"]["message"].startswith("unknown service")
    assert api.delete("/api/models/whisper-large-v3/register").status_code == 204
    assert api.get("/api/models").json() == []


def test_lease_lifecycle_over_http(api, cat):
    r = api.post("/api/leases", json={"holder": "bench:job-1", "models": ["default-asr"]})
    assert r.status_code == 201 and r.json()["state"] == "starting"
    lid = r.json()["id"]
    step(cat)
    assert api.get(f"/api/leases/{lid}").json()["state"] == "active"
    assert [x["id"] for x in api.get("/api/leases").json()] == [lid]
    assert api.post(f"/api/leases/{lid}/heartbeat").json()["state"] == "active"
    assert api.get("/api/models/whisper-large-v3").json()["leased_by"] == ["bench:job-1"]
    assert api.delete(f"/api/leases/{lid}").status_code == 204
    assert api.post(f"/api/leases/{lid}/heartbeat").status_code == 409
    assert api.post("/api/leases", json={"holder": "x", "models": ["nope"]}).status_code == 404


def test_one_gpu_model_per_lease_and_the_swap_count(api, cat):
    two = api.post("/api/leases", json={"holder": "x", "models": ["default-asr", "default-embed", "silero-vad"]})
    assert two.status_code == 422 and "lease one GPU model per stage" in two.json()["error"]["message"]
    assert api.get("/api/platform").json()["swaps"] == 0
    cat.settings.release_linger_s = 3600.0
    for model in ["whisper-large-v3", "bge-m3"]:
        lid = api.post("/api/leases", json={"holder": "x", "models": [model]}).json()["id"]
        step(cat)
        assert api.get(f"/api/leases/{lid}").json()["state"] == "active"
        api.delete(f"/api/leases/{lid}")
    gpu = api.get("/api/platform").json()
    assert gpu["swaps"] == 1 and gpu["gpu_models"] == [{"id": "bge-m3", "fraction": 0.93}]  # whole GPU, not 0.08


def test_aicore_client_against_the_api(api, cat):
    c = Catalog("http://testserver", http=api)
    assert c.available()
    c.register("silero-vad", tags=["vad"])
    assert [m["id"] for m in c.models(tag="vad")] == ["silero-vad"]
    lease = c.request_lease("label:job-1", ["silero-vad"])
    step(cat)
    assert c.get_lease(lease["id"])["state"] == "active"
    c.release(lease["id"])


def test_output_index_across_apps(api):
    api.post("/api/apps", json={"name": "bench", "base_url": "http://bench:8000/"})
    key = index_key("b" * 16, "f" * 64, 1.23456, 2.5, "p" * 16)
    whole = index_key("b" * 16, "f" * 64, None, None, "p" * 16)
    assert api.post("/api/index/record", json={"app": "bench", "entries": [{**key, "output_id": 812},
                                                                           {**whole, "output_id": 7}]}).json() == {"recorded": 2}
    hits = api.post("/api/index/lookup", json={"app": "label", "keys": [key, whole, {**key, "params_hash": "q" * 16}]}).json()["hits"]
    assert hits == [{"owner": "bench", "output_id": 812, "owner_url": "http://bench:8000"},
                    {"owner": "bench", "output_id": 7, "owner_url": "http://bench:8000"}, None]
    api.post("/api/index/record", json={"app": "label", "entries": [{**key, "output_id": 3}]})
    assert api.post("/api/index/lookup", json={"app": "label", "keys": [key]}).json()["hits"][0]["owner"] == "label"
    assert api.post("/api/index/forget", json={"app": "bench", "output_ids": [812, 7]}).json() == {"forgotten": 2}
    bad = api.post("/api/index/record", json={"app": "bench", "entries": [{**key, "build_key": None, "output_id": 1}]})
    assert bad.status_code == 422


def test_staged_plan_loads_each_model_once_and_skips_indexed_work(api, cat):
    wait_builds(cat, ["pyannote-diarization", "silero-vad"])
    bk = cat.scheduler.runs["pyannote-diarization"].build.build_key
    keys = [index_key("x", f"{i:064x}", None, None, "p" * 16) for i in range(3)]
    api.post("/api/index/record", json={"app": "bench", "entries": [{**k, "build_key": bk, "output_id": i}
                                                                     for i, k in enumerate(keys)]})
    body = {"stages": [{"name": "seg", "models": ["pyannote-diarization", "silero-vad"],
                        "keys": {"pyannote-diarization": keys, "silero-vad": keys}},
                       {"name": "asr", "models": ["whisper-large-v3", "qwen3-asr-1-7b"], "after": "seg"}]}
    plan = api.post("/api/jobs", json=body).json()
    steps = {s["model"]: s for s in plan["steps"]}
    assert [s["stage"] for s in plan["steps"]] == ["seg", "seg", "asr", "asr"]
    assert steps["pyannote-diarization"]["skip"] and not steps["pyannote-diarization"]["load"]
    assert steps["silero-vad"] == {**steps["silero-vad"], "items": 3, "hits": 0, "missing": 3, "load": True}
    assert plan["loads"] == 3 and plan["hits"] == 3
    co = api.post("/api/jobs", json={**body, "pipeline": "co-resident"}).json()
    assert co["steps"][0]["stage"] == "asr"  # the slowest loads (vLLM) are the outer loop
    assert api.post("/api/jobs", json={"stages": [{"name": "a", "models": ["x"], "after": "zz"}]}).status_code == 422


def test_gateway_forwards_unchanged_and_counts_calls(api, cat, monkeypatch):
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "hi"}, headers={"x-model": "whisper"})

    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    r0 = cat.scheduler.request("bench:job-1", ["whisper-large-v3"])
    step(cat)
    assert cat.scheduler.get(r0.id).state == "active"
    r = api.post("/m/default-asr/v1/audio/transcriptions?x=1", files={"file": ("a.wav", b"RIFF....", "audio/wav")},
                 data={"model": "whisper-large-v3"})
    assert r.status_code == 200 and r.json() == {"text": "hi"} and r.headers["x-model"] == "whisper"
    assert str(seen[0].url) == "http://localhost:8101/v1/audio/transcriptions?x=1"
    assert b"RIFF...." in seen[0].content
    step(cat, 1)
    assert cat.scheduler.runs["whisper-large-v3"].in_flight == 0
    assert api.get("/api/usage").json()[[u["id"] for u in api.get("/api/usage").json()].index("whisper-large-v3")]["calls"] == 1
    cat.settings.gateway_unmeasured_wait_s = 0.3  # no scheduler threads here: the start never happens
    oip = api.post("/v2/models/silero-vad/infer", json={"inputs": []})
    assert oip.status_code == 503 and oip.json()["error"]["type"] == "model_starting" and oip.headers["retry-after"]
    assert api.get("/m/nope/v1/models").status_code == 404


def test_gateway_starts_a_stopped_model_and_holds_the_call(api, cat):
    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": 1})))
    cat.scheduler.start()  # real threads: the call waits while the starter brings the model up
    try:
        r = api.get("/m/silero-vad/v2/models/silero-vad/ready")
        assert r.status_code == 200 and r.json() == {"ok": 1}
        assert cat.scheduler.is_up("silero-vad")
    finally:
        cat.scheduler.shutdown()
