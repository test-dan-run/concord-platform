"""The hosted connection test: one line per step, a time on every result, and no 500 on a bad provider answer."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}
OIP = {**ASR, "id": "remote-vad", "contract": "oip-v2", "api": "infer", "base_url": "https://inference.example.com",
       "model": "vad-x"}


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


def provider(cat, handler):
    def wrapped(r):
        r.read()
        return handler(r)

    cat.hosted.transport = httpx.MockTransport(wrapped)


def statuses(out):
    return [(x["name"], x["status"]) for x in out["lines"]]


def test_a_timeout_after_the_host_answered_is_the_contract_step(api, cat):
    def handler(r):
        if r.url.path == "/v1/models":
            return httpx.Response(404)
        raise httpx.ReadTimeout("timed out", request=r)

    provider(cat, handler)
    out = api.post("/api/hosted/test", json=ASR).json()
    assert statuses(out) == [("tls", "pass"), ("key", "warn"), ("contract", "fail")], out
    assert "ReadTimeout" in out["lines"][2]["reason"] and not out["ok"] and out["tested_at"]


def test_no_models_route_leaves_the_key_unchecked_until_a_request_is_accepted(api, cat):
    provider(cat, lambda r: httpx.Response(404) if r.url.path == "/v1/models" else httpx.Response(200, json={"text": ""}))
    out = api.post("/api/hosted/test", json=ASR).json()
    assert statuses(out) == [("tls", "pass"), ("key", "warn"), ("contract", "pass")] and out["ok"], out
    assert "404" in out["lines"][1]["reason"]


def test_oip_metadata_that_is_not_json_is_a_contract_failure_not_a_500(api, cat):
    provider(cat, lambda r: httpx.Response(200, json={"ready": True}) if r.url.path.endswith("/ready")
             else httpx.Response(200, text="<html>login</html>"))
    t = api.post("/api/hosted/test", json=OIP)
    assert t.status_code == 200 and statuses(t.json()) == [("tls", "pass"), ("key", "pass"), ("contract", "fail")]
    assert "<html>login</html>" in t.json()["lines"][2]["reason"] and t.json()["signature"] is None
    c = api.post("/api/hosted", json=OIP)
    assert c.status_code == 422 and "cannot read vad-x's OIP metadata" in c.json()["error"]["message"]


def test_every_result_has_its_test_time(api, cat):
    provider(cat, lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    assert api.post("/api/hosted", json=ASR).status_code == 201

    def handler(r):
        raise httpx.ConnectError("refused", request=r)

    provider(cat, handler)
    out = api.post("/api/models/hosted-asr/test").json()
    assert not out["ok"] and out["tested_at"]
    assert api.get("/api/models/hosted-asr").json()["hosted"]["last_test"]["tested_at"] == out["tested_at"]
