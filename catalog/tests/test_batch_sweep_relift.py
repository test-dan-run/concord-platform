"""An out of memory split at one length of the sweep must not leave the server's limit lowered for the next
lengths. The fake device below runs the real aisvc MicroBatcher and set_limits: it holds 9.6 GiB, so a batch
of 128 x 2 s (256 s padded) runs out of memory and is split, while 16 x 15 s (240 s padded, 9.4 GiB) fits."""

from __future__ import annotations

import json

import httpx
import pytest
from aisvc import batch as ab

from catalog import batching
from catalog.batching import CONTEXT_GIB, LIFT, Memory, sweep

LIMIT, BASE, C = 9.6, 1.0, 0.035  # GiB the device holds, GiB at rest, GiB per second of padded audio


def true_peak(n: int, length_s: float) -> float:
    return BASE + C * n * length_s


@pytest.fixture
def device(monkeypatch):
    """A MicroBatcher on a fake GPU, its limits lifted like the profiler does, and the server's memory routes."""
    monkeypatch.setattr(ab, "_LIMITS", {})
    state = {"peak": 0.0}

    def fn(items):
        need = BASE + C * len(items) * max(len(x) for x in items) / 1000.0  # an item is a list of ms
        if need > LIMIT:
            raise RuntimeError("CUDA out of memory (fake)")
        state["peak"] = max(state["peak"], need)
        return [0] * len(items)

    mb = ab.MicroBatcher(fn, length=len, name="embed")

    def send(n, length_s):
        mb([[0] * int(length_s * 1000)] * n)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/platform/batch-limits":
            body = json.loads(request.content)
            return httpx.Response(200, json=ab.set_limits(body.get("max_items"), body.get("max_len"),
                                                          restore=bool(body.get("restore"))))
        if path == "/platform/memory/reset":
            state["peak"] = 0.0
            return httpx.Response(200, json={"cuda": True, "reserved_gib": 0.0, "batch": ab.limits_stats()})
        return httpx.Response(200, json={"cuda": True, "peak_reserved_gib": state["peak"], "batch": ab.limits_stats()})

    return send, handler


def test_the_sweep_lifts_the_limits_again_before_each_length(device):
    send, handler = device
    http = httpx.Client(transport=httpx.MockTransport(handler))
    http.post("http://svc/platform/batch-limits", json=LIFT)
    lifts = []

    def relift():
        lifts.append(1)
        http.post("http://svc/platform/batch-limits", json=LIFT)

    pts, stops, _ = sweep(send, Memory("http://svc", http, lambda: None, None), [2.0, 15.0], budget_gib=None,
                          max_n=128, relift=relift)
    assert "out of memory at 128" in stops[0]["stopped"] and lifts == [1]
    at15 = {p.n: p.peak_gib for p in pts if p.length_s == 15.0}
    assert at15[16] == pytest.approx(true_peak(16, 15.0) + CONTEXT_GIB, abs=1e-3)  # not 5.2 GiB, split by 8
    assert 32 not in at15 and "out of memory at 32" in stops[1]["stopped"]


def test_a_measurement_lifts_the_limits_again_after_a_split(cat, device, monkeypatch):
    send, handler = device
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda timeout: real(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(batching, "sender", lambda meta, url, http: send)
    p = cat.scheduler.platform()
    s = p.get("wavlm-sv")  # lengths 2, 15 and 20 s (max_item_s 20)
    prof = cat.profiler._measure(p, s, {}, "rid", None, 100.0)
    at15 = {pt.n: pt.peak_gib for pt in prof.points if pt.length_s == 15.0}
    assert at15[16] == pytest.approx(true_peak(16, 15.0) + CONTEXT_GIB, abs=1e-3)
    assert all(lim.max_len == lim.configured[1] for lim in ab._LIMITS.values())  # restored at the end
