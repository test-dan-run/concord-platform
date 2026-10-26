"""The gateway passes headers on as the model or provider sent them: a header sent twice (two Set-Cookie) stays two,
and a value keeps its bytes, also when they are UTF-8."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = "sk-test-0123456789abcdef7f3a"
ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": KEY, "timeout_s": 5, "max_concurrency": 2,
       "max_upload_mb": 1, "data_leaves_network": True}
NAME = "café ☕".encode()  # é has a latin-1 code point, ☕ has none
COOKIES = [(b"Set-Cookie", b"a=1; Path=/"), (b"Set-Cookie", b"b=2; Path=/")]


@pytest.fixture
def app(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    cat.scheduler.is_up = lambda name: True
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        assert c.post("/api/hosted", json=ASR).status_code == 201
        yield app


def answer(cat, status: int) -> None:
    def reply(r: httpx.Request) -> httpx.Response:
        r.read()
        return httpx.Response(status, headers=[*COOKIES, (b"X-File-Name", NAME), (b"X-Echo", f"key {KEY}".encode()),
                                               (b"Connection", b"close")], stream=httpx.ByteStream(b'{"ok": true}'))

    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))


def get(app, path: str) -> httpx.Response:
    """GET through the ASGI app, keeping the header bytes it sends (the TestClient decodes them as ASCII)."""
    async def call() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
            return await c.get(path)

    return asyncio.run(call())


# A platform model's reply is streamed whatever its status; a provider's is streamed (200) or read and redacted (400).
@pytest.mark.parametrize("status, path", [(200, "/m/silero-vad/v2/models/silero-vad/ready"),
                                          (200, "/m/hosted-asr/v1/models"), (400, "/m/hosted-asr/v1/models")])
def test_headers_reach_the_app_as_sent(app, cat, path, status):
    answer(cat, status)
    r = get(app, path)
    assert r.status_code == status and r.json() == {"ok": True}
    assert r.headers.get_list("set-cookie") == ["a=1; Path=/", "b=2; Path=/"]
    raw = dict(r.headers.raw)
    assert raw[b"x-file-name"] == NAME and b"connection" not in raw  # hop-by-hop headers are still dropped
    # A provider's error reply drops a header that quotes the key; a platform model's reply is passed on.
    assert (b"x-echo" in raw) == (status < 400 or "silero" in path)
