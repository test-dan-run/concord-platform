"""Hosted keys stay out of what the catalog stores and shows, even when a provider echoes them back."""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.verifier import probes

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}


def echoing_provider(r: httpx.Request) -> httpx.Response:
    """Accepts the key on /v1/models; rejects verbose_json quoting the key it got, and puts it in a plain transcript."""
    r.read()
    got = r.headers.get("authorization", "")
    if r.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "whisper-1"}]})
    if b"verbose_json" in r.content:
        return httpx.Response(400, json={"error": f"request rejected for credential {got}"})
    return httpx.Response(200, json={"text": f"you sent {got}"})


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_verification_report_and_record_hide_a_key_the_provider_echoes(api, cat):
    cat.hosted.transport = httpx.MockTransport(echoing_provider)
    assert api.post("/api/hosted", json=ASR).status_code == 201
    cat.verifier.pending.clear()
    cat.verifier.run_hosted(cat.hosted.get("hosted-asr"))
    rec = api.get("/api/models/hosted-asr")
    assert KEY not in rec.text and KEY not in api.get("/api/models").text
    report = rec.json()["capabilities"]["report"]
    assert any("••••" in x["reason"] for x in report["lines"]), report["lines"]
    assert report["sample"]["transcript"] == "you sent Bearer ••••"
    row = cat.db.one("SELECT record, report FROM capability WHERE model_id = 'hosted-asr'")
    assert KEY not in row["record"] and KEY not in row["report"]


def test_a_crashed_hosted_verification_hides_the_key_in_the_record_and_the_log(api, cat, monkeypatch, caplog):
    cat.hosted.transport = httpx.MockTransport(echoing_provider)
    assert api.post("/api/hosted", json=ASR).status_code == 201
    cat.verifier.pending.clear()

    def crash(meta, base_url, http=None):
        raise RuntimeError(f"provider said: bad credential {KEY}")

    monkeypatch.setattr(probes, "verify", crash)
    with caplog.at_level(logging.INFO, logger="catalog.verifier"):
        cat.verifier.run("hosted-asr")  # what the verifier's thread does
    lines = api.get("/api/models/hosted-asr").json()["capabilities"]["report"]["lines"]
    assert lines == [{"name": "verification", "status": "fail",
                      "reason": "verification crashed: provider said: bad credential ••••"}]
    assert "bad credential ••••" in caplog.text and KEY not in caplog.text
