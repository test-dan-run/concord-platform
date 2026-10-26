"""The sweep of an OpenAI transcription server sends no more concurrent requests than the server runs at once:
aisvc's transcriptions route is a sync function that Starlette runs in a pool of 40 threads."""

from __future__ import annotations

import httpx

from catalog import batching
from tools import render


def test_an_openai_asr_sweep_sends_at_most_32_at_once(cat, monkeypatch):
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cuda": True, "reserved_gib": 0.0, "peak_reserved_gib": 4.0, "batch": {}})

    monkeypatch.setattr(httpx, "Client", lambda timeout: real(transport=httpx.MockTransport(handler)))
    sent: list[int] = []
    monkeypatch.setattr(batching, "sender", lambda meta, url, http: lambda n, length: sent.append(n))
    p = cat.scheduler.platform()
    s = p.get("ark-asr-0-6b")
    meta = render.endpoint_entry(p, s)
    assert batching.mode(meta) == "openai-asr"
    prof = cat.profiler._measure(p, s, meta, "rid", None, 100.0)
    assert max(sent) == 32 and max(pt.n for pt in prof.points) == 32
