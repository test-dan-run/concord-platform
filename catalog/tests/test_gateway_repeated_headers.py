"""The gateway keeps a header the model or provider sends twice (two Set-Cookie headers stay two)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

ASR = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions", "base_url": "https://api.example.com/v1",
       "model": "whisper-1", "auth_type": "bearer", "key": "sk-test-0123456789abcdef7f3a", "timeout_s": 5,
       "max_concurrency": 2, "max_upload_mb": 1, "data_leaves_network": True}
COOKIES = [("set-cookie", "a=1; Path=/"), ("set-cookie", "b=2; Path=/")]


@pytest.fixture
def api(cat):
    cat.hosted.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "whisper-1"}]}))
    cat.scheduler.is_up = lambda name: True
    with TestClient(create_app(cat, start=False)) as c:
        assert c.post("/api/hosted", json=ASR).status_code == 201
        yield c


def answer(cat, status: int) -> None:
    def reply(r: httpx.Request) -> httpx.Response:
        r.read()
        return httpx.Response(status, headers=[*COOKIES, ("x-trace", "t1"), ("connection", "close")],
                              stream=httpx.ByteStream(b'{"ok": true}'))

    cat.gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))


@pytest.mark.parametrize("path", ["/m/silero-vad/v2/models/silero-vad/ready", "/m/hosted-asr/v1/models"])
@pytest.mark.parametrize("status", [200, 400])
def test_a_repeated_header_reaches_the_app_twice(api, cat, path, status):
    answer(cat, status)
    r = api.get(path)
    assert r.status_code == status and r.json() == {"ok": True}
    assert r.headers.get_list("set-cookie") == ["a=1; Path=/", "b=2; Path=/"]
    assert r.headers["x-trace"] == "t1" and "connection" not in r.headers  # hop-by-hop headers are still dropped
