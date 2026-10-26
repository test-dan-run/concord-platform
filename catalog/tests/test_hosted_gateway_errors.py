"""The gateway redacts the key in a hosted provider's error reply and streams a successful reply unchanged."""

import gzip

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "tags": ["asr"], "data_leaves_network": True}


def models(r: httpx.Request) -> httpx.Response:
    r.read()
    return httpx.Response(200, json={"data": [{"id": "whisper-1"}]})


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(models)  # the connection test and verification on create
    with TestClient(create_app(cat, start=False)) as c:
        assert c.post("/api/hosted", json=ASR).status_code == 201
        yield c


def answer(cat, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
    """The provider answers every gateway call with this reply, streamed (not read in advance)."""
    def reply(r: httpx.Request) -> httpx.Response:
        r.read()
        return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))

    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))


def call(api) -> httpx.Response:
    return api.post("/m/hosted-asr/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "whisper-1"})


def test_an_error_reply_that_echoes_the_key_is_redacted(api, cat):
    body = gzip.compress(f'{{"error": {{"message": "bad request for key {KEY}"}}}}'.encode())
    answer(cat, 400, body, {"content-type": "application/json", "content-encoding": "gzip",
                            "content-length": str(len(body)), "x-request-key": f"Bearer {KEY}"})
    r = call(api)
    assert r.status_code == 400 and KEY not in r.text and KEY not in str(r.headers)
    assert r.json() == {"error": {"message": "bad request for key ••••"}}  # decompressed, then redacted
    assert "x-request-key" not in r.headers  # a header cannot carry the mask, so it is dropped
    assert "content-encoding" not in r.headers  # sent uncompressed
    assert r.headers["content-type"] == "application/json" and int(r.headers["content-length"]) == len(r.content)
    assert api.get("/api/models/hosted-asr").json()["in_flight"] == 0


def test_an_error_reply_over_1_mib_is_not_forwarded(api, cat):
    answer(cat, 500, b"x" * (1024 * 1024) + KEY.encode())
    r = call(api)
    assert r.status_code == 500 and KEY not in r.text
    assert r.json()["error"]["type"] == "upstream_error" and "over 1 MiB" in r.json()["error"]["message"]
    assert api.get("/api/models/hosted-asr").json()["in_flight"] == 0


def test_a_successful_reply_is_streamed_unchanged(api, cat):
    raw = gzip.compress(f'{{"text": "you sent {KEY}"}}'.encode())
    answer(cat, 200, raw, {"content-type": "application/json", "content-encoding": "gzip",
                           "content-length": str(len(raw))})
    with api.stream("POST", "/m/hosted-asr/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "whisper-1"}) as r:
        assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
        got = b"".join(r.iter_raw())
    assert got == raw and r.headers["content-length"] == str(len(raw))
    assert gzip.decompress(got) == f'{{"text": "you sent {KEY}"}}'.encode()  # a successful reply is not redacted


def test_an_error_reply_that_breaks_off_is_a_502_without_the_key(api, cat):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield f"bad key {KEY}".encode()
            raise httpx.ReadTimeout("timed out")

    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, stream=Broken())))
    r = call(api)
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_error" and KEY not in r.text
    assert api.get("/api/models/hosted-asr").json()["in_flight"] == 0
