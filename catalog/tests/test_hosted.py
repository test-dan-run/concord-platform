"""Hosted endpoints: encrypted keys, the gateway's hosted branch, leases, build keys and verification,
against a fake provider (an httpx mock), so nothing leaves the machine."""

import time

import httpx
import pytest
from catalog_helpers import step
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.verifier import probes

KEY = "sk-test-0123456789abcdef7f3a"
TRUTH = probes.clip()[1]
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr", "hosted"], "data_leaves_network": True}
OIP_SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
                      {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]}],
           "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}]}


class Provider:
    """A fake provider: OpenAI ASR under /v1, OIP under /v2, a bearer or custom-header key."""

    def __init__(self, key=KEY, header="authorization"):
        self.key, self.header = key, header
        self.seen: list[httpx.Request] = []

    def authorised(self, r: httpx.Request) -> bool:
        got = r.headers.get(self.header, "")
        return got == (f"Bearer {self.key}" if self.header == "authorization" else self.key)

    def __call__(self, r: httpx.Request) -> httpx.Response:
        r.read()
        self.seen.append(r)
        if not self.authorised(r):
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        path = r.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "whisper-1", "object": "model"},
                                                                          {"id": "chat-x", "object": "model"}]})
        if path == "/v1/chat/completions":
            if b"input_audio" in r.content:
                return httpx.Response(400, json={"error": {"message": "audio input is not supported"}})
            return httpx.Response(200, json={"id": "c1", "object": "chat.completion", "created": 1, "model": "chat-x",
                                             "choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"role": "assistant", "content": "hello"}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        if path == "/v1/audio/transcriptions":
            text = TRUTH["transcript"] if len(r.content) > 100_000 else "a tone"
            if b"verbose_json" in r.content:
                return httpx.Response(200, json={"text": text, "segments": [{"id": 0, "start": 0.4, "end": 9.9, "text": text}]})
            return httpx.Response(200, json={"text": text})
        if path == "/v2/models/vad-x/ready":
            return httpx.Response(200, json={"name": "vad-x", "ready": True})
        if path == "/v2/models/vad-x":
            return httpx.Response(200, json={"name": "vad-x", "platform": "x", **OIP_SIG})
        if path == "/v2/models/vad-x/infer":
            return httpx.Response(200, json={"model_name": "vad-x", "outputs": []})
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


@pytest.fixture
def provider(cat):
    p = Provider()
    cat.hosted.transport = httpx.MockTransport(p)
    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(p))
    return p


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_create_stores_the_key_encrypted_and_never_returns_it(api, cat, provider, settings):
    r = api.post("/api/hosted", json=ASR)
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["state"] == "hosted" and rec["gpu"] is None and rec["image"] is None and rec["served_name"] == "whisper-1"
    assert rec["hosted"]["key_hint"] == "7f3a" and rec["hosted"]["resolved_url"] == "https://api.example.com/v1/audio/transcriptions"
    assert rec["capabilities"]["status"] == "pending" and rec["tags"] == ["asr", "hosted"]
    assert KEY not in r.text and KEY not in api.get("/api/models").text and KEY not in api.get("/api/models/hosted-asr").text
    assert [m["id"] for m in api.get("/api/models", params={"tag": "hosted"}).json()] == ["hosted-asr"]
    cat.db._conn.execute("PRAGMA wal_checkpoint(FULL)")
    raw = b"".join(f.read_bytes() for f in settings.db_path.parent.glob("catalog.db*"))
    assert KEY.encode() not in raw and b"7f3a" in raw  # only the hint is in the clear


def test_create_is_refused_without_the_data_confirmation_or_with_a_taken_id(api, provider):
    assert api.post("/api/hosted", json={**ASR, "data_leaves_network": False}).status_code == 422
    assert api.post("/api/hosted", json={**ASR, "id": "whisper-large-v3"}).status_code == 409
    assert api.post("/api/hosted", json={**ASR, "id": "default-asr"}).status_code == 409  # an alias
    assert api.post("/api/hosted", json={**ASR, "key": None}).status_code == 422
    assert api.post("/api/hosted", json={**ASR, "contract": "openai", "api": "completions"}).status_code == 422
    assert api.post("/api/hosted", json=ASR).status_code == 201
    assert api.post("/api/hosted", json=ASR).status_code == 409


def test_gateway_adds_the_stored_key_and_removes_the_callers_auth(api, provider):
    api.post("/api/hosted", json=ASR)
    provider.seen.clear()
    r = api.post("/m/hosted-asr/v1/audio/transcriptions", headers={"Authorization": "Bearer app-secret"},
                 files={"file": ("a.wav", b"RIFF....", "audio/wav")}, data={"model": "whisper-1"})
    assert r.status_code == 200 and r.json()["text"] == "a tone"
    sent = provider.seen[-1]
    assert str(sent.url) == "https://api.example.com/v1/audio/transcriptions"
    assert sent.headers["authorization"] == f"Bearer {KEY}" and b"RIFF...." in sent.content and b"whisper-1" in sent.content
    assert [u for u in api.get("/api/usage").json() if u["id"] == "hosted-asr"][0]["calls"] == 0  # flushed on a tick


def test_custom_header_auth_strips_a_header_of_the_same_name(api, cat):
    p = Provider(header="api-key")
    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(p))
    cat.hosted.transport = httpx.MockTransport(p)
    assert api.post("/api/hosted", json={**ASR, "auth_type": "header", "header_name": "api-key"}).status_code == 201
    r = api.get("/m/hosted-asr/v1/models", headers={"api-key": "from-the-app", "authorization": "Bearer x"})
    assert r.status_code == 200
    assert p.seen[-1].headers["api-key"] == KEY and "authorization" not in p.seen[-1].headers


def test_upload_limit_is_enforced_before_sending(api, provider):
    api.post("/api/hosted", json=ASR)  # 1 MB
    provider.seen.clear()
    big = b"x" * (1024 * 1024 + 1)
    r = api.post("/m/hosted-asr/v1/audio/transcriptions", content=big, headers={"content-type": "application/octet-stream"})
    assert r.status_code == 413 and r.json()["error"]["type"] == "payload_too_large" and provider.seen == []

    def chunks():  # no Content-Length: counted while streaming
        for _ in range(3):
            yield b"y" * 600_000

    r = api.post("/m/hosted-asr/v1/audio/transcriptions", content=chunks(), headers={"content-type": "application/octet-stream"})
    assert r.status_code == 413 and provider.seen == []


def test_concurrency_limit_waits_then_503(api, cat, provider):
    import asyncio

    api.post("/api/hosted", json={**ASR, "max_concurrency": 1, "timeout_s": 0.2})
    cat.gateway._slots["hosted-asr"] = (1, asyncio.Semaphore(0))  # the one slot is taken
    r = api.get("/m/hosted-asr/v1/models")
    assert r.status_code == 503 and r.json()["error"]["type"] == "concurrency_limit"
    cat.gateway._slots.clear()
    assert api.get("/m/hosted-asr/v1/models").status_code == 200


def test_build_key_changes_with_the_endpoint_not_with_limits_or_key(api, cat, provider):
    rec = api.post("/api/hosted", json=ASR).json()
    cat.verifier.pending.clear()
    cat.verifier._order.clear()
    same = api.patch("/api/hosted/hosted-asr", json={"max_concurrency": 8, "timeout_s": 60}).json()
    assert same["build_key"] == rec["build_key"] and same["hosted"]["max_concurrency"] == 8
    assert cat.verifier.status("hosted-asr") is None
    provider.key = "sk-new-key-0000"
    assert api.patch("/api/hosted/hosted-asr", json={"key": "sk-wrong"}).status_code == 422  # tested before replacing
    replaced = api.patch("/api/hosted/hosted-asr", json={"key": "sk-new-key-0000"}).json()
    assert replaced["build_key"] == rec["build_key"] and replaced["hosted"]["key_hint"] == "0000"
    assert cat.verifier.status("hosted-asr") is None
    moved = api.patch("/api/hosted/hosted-asr", json={"base_url": "https://eu.api.example.com/v1/"}).json()
    assert moved["build_key"] != rec["build_key"] and cat.verifier.status("hosted-asr") == "pending"


def test_leases_grant_hosted_models_at_once_and_start_stop_refuse(api, cat, provider):
    api.post("/api/hosted", json=ASR)
    r = api.post("/api/leases", json={"holder": "bench:job-1", "models": ["hosted-asr"]})
    assert r.status_code == 201 and r.json()["state"] == "active"
    # A hosted endpoint, a CPU model and one GPU model are granted together.
    mixed = api.post("/api/leases", json={"holder": "bench:job-2",
                                          "models": ["hosted-asr", "silero-vad", "whisper-large-v3"]}).json()
    assert mixed["state"] == "starting" and mixed["models"] == ["hosted-asr", "silero-vad", "whisper-large-v3"]
    step(cat)
    assert api.get(f"/api/leases/{mixed['id']}").json()["state"] == "active"
    assert api.get("/api/models/hosted-asr").json()["leased_by"] == ["bench:job-1", "bench:job-2"]
    assert api.post("/api/models/hosted-asr/stop").status_code == 409
    assert api.post("/api/models/hosted-asr/start").status_code == 409
    step(cat)
    assert api.get(f"/api/leases/{r.json()['id']}").json()["state"] == "active"  # not dropped as unknown


def test_connection_test_checks_tls_key_and_contract(api, provider):
    ok = api.post("/api/hosted/test", json=ASR).json()
    assert ok["ok"] and [x["name"] for x in ok["lines"]] == ["tls", "key", "contract"]
    bad = api.post("/api/hosted/test", json={**ASR, "key": "sk-nope"}).json()
    assert not bad["ok"] and bad["lines"][1]["status"] == "fail" and "sk-nope" not in str(bad)
    other = api.post("/api/hosted/test", json={**ASR, "model": "gpt-x"}).json()
    assert not other["ok"] and "not 'gpt-x'" in other["lines"][-1]["reason"]
    api.post("/api/hosted", json=ASR)
    assert api.post("/api/models/hosted-asr/test").json()["ok"]
    assert api.post("/api/models/silero-vad/test").status_code == 409


def test_verification_probes_the_provider_with_its_model_name(api, cat, provider):
    api.post("/api/hosted", json=ASR)
    cat.verifier.pending.clear()  # run it here instead of on the verifier's thread
    cat.verifier.run_hosted(cat.hosted.get("hosted-asr"))
    caps = api.get("/api/models/hosted-asr").json()["capabilities"]
    assert caps["status"] == "verified" and caps["usable_as"] == "asr" and caps["timing_level"] == "segment", caps["report"]
    bodies = [r.content for r in provider.seen if r.url.path == "/v1/audio/transcriptions"]
    assert bodies and all(b"whisper-1" in b for b in bodies)
    assert not any(r.url.path == "/health" for r in provider.seen)  # no platform health route at a provider
    assert api.get("/api/models/hosted-asr").json()["hosted"]["next_check_at"]


def test_daily_reverify_queues_hosted_models_verified_over_a_day_ago(api, cat, provider):
    api.post("/api/hosted", json=ASR)
    cat.verifier.run_hosted(cat.hosted.get("hosted-asr"))
    cat.verifier.pending.clear()
    cat.verifier._order.clear()
    cat.verifier.scan()
    assert cat.verifier.status("hosted-asr") is None
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 25 * 3600))
    cat.db.execute("UPDATE capability SET verified_at = ? WHERE model_id = 'hosted-asr'", (old,))
    cat.verifier.scan()
    assert cat.verifier.status("hosted-asr") == "pending"


def test_oip_hosted_takes_its_signature_and_rewrites_the_model_in_the_path(api, provider):
    body = {**ASR, "id": "remote-vad", "contract": "oip-v2", "api": "infer", "base_url": "https://inference.example.com",
            "model": "vad-x"}
    rec = api.post("/api/hosted", json=body).json()
    assert rec["signature"] == OIP_SIG and rec["hosted"]["resolved_url"] == "https://inference.example.com/v2/models/vad-x/infer"
    assert api.post("/v2/models/remote-vad/infer", json={"inputs": []}).status_code == 200
    assert str(provider.seen[-1].url) == "https://inference.example.com/v2/models/vad-x/infer"
    assert api.post("/m/remote-vad/v2/models/vad-x/infer", json={"inputs": []}).status_code == 200  # aicore's form
    assert str(provider.seen[-1].url) == "https://inference.example.com/v2/models/vad-x/infer"
    from catalog.verifier import _with_examples

    assert _with_examples(OIP_SIG)["inputs"][1]["example"] == [16000]  # metadata has none; conformance needs one


def test_missing_master_key_keeps_the_catalog_up_and_hosted_calls_get_503(api, cat, provider, master_key):
    api.post("/api/hosted", json=ASR)
    master_key.unlink()
    cat.hosted.keys._fernet = None
    r = api.post("/api/hosted", json={**ASR, "id": "hosted-2"})
    assert r.status_code == 503 and str(master_key) in r.json()["error"]["message"]
    call = api.get("/m/hosted-asr/v1/models")
    assert call.status_code == 503 and call.json()["error"]["type"] == "key_unavailable"
    assert api.get("/api/health").status_code == 200
    assert "key unavailable" in api.get("/api/models/hosted-asr").json()["state_detail"]


def test_delete_removes_the_endpoint_its_key_and_its_registration(api, cat, provider):
    api.post("/api/hosted", json=ASR)
    assert api.delete("/api/hosted/hosted-asr").status_code == 204
    assert api.get("/api/models/hosted-asr").status_code == 404 and api.get("/api/models").json() == []
    assert cat.db.one("SELECT * FROM hosted") is None and cat.db.one("SELECT * FROM registration") is None
    api.post("/api/hosted", json=ASR)
    assert api.delete("/api/models/hosted-asr/register").status_code == 204  # Unregister removes it whole
    assert cat.db.one("SELECT * FROM hosted") is None


def test_hosted_chat_without_audio_is_verified_as_a_text_model(api, cat, provider):
    api.post("/api/hosted", json={**ASR, "id": "remote-chat", "api": "chat.completions", "model": "chat-x"})
    cat.verifier.pending.clear()
    cat.verifier.run_hosted(cat.hosted.get("remote-chat"))
    caps = api.get("/api/models/remote-chat").json()["capabilities"]
    assert caps["status"] == "verified" and caps["usable_as"] is None, caps["report"]
    assert any(x["name"] == "audio" and x["status"] == "info" for x in caps["report"]["lines"])
