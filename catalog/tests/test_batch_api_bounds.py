"""Batch API query values that overflow the recommendation (500 before) are a 422. The audio limit that rounds to
0 ms is checked in test_batching.py."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_batching import TRUE, points

from catalog.app import create_app
from catalog.batching import Profile

# One value per declared bound: no inf or nan, and each parameter's upper (and item_s's lower) limit.
AUDIO = [{"vram_gb": "inf", "item_s": 5}, {"vram_gb": 1e308, "item_s": 5}, {"vram_gb": 24, "item_s": "inf"},
         {"vram_gb": 24, "item_s": 1e-310}, {"vram_gb": 24, "item_s": 1e300}, {"vram_gb": 24, "item_s": 5, "n": 10**400}]
KV = [{"item_tokens": 10**400}]


@pytest.fixture
def profiled(cat, monkeypatch):
    audio = Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=None, points=points())
    kv = Profile(kind="kv", kv_tokens=200_000, kv_gib=20.0, fixed_gib=5.0)
    cat.profiler._save("wavlm-sv", "bk-audio", "measured", audio, None)
    cat.profiler._save("qwen3-8b", "bk-kv", "measured", kv, None)
    monkeypatch.setattr(cat.builds, "current",
                        lambda p, s, c: type("B", (), {"build_key": "bk-kv" if s.name == "qwen3-8b" else "bk-audio"})())
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c, audio, kv


@pytest.mark.parametrize("model,params", [("wavlm-sv", x) for x in AUDIO] + [("qwen3-8b", x) for x in KV])
def test_a_non_finite_or_absurd_query_value_is_a_422(profiled, model, params):
    c, _, _ = profiled
    assert c.get(f"/api/models/{model}/batch", params=params).status_code == 422


def test_ordinary_query_values_still_work(profiled):
    c, audio, kv = profiled
    r = c.get("/api/models/wavlm-sv/batch", params={"vram_gb": 24, "item_s": 5, "n": 8})
    assert r.status_code == 200 and r.json()["recommendation"]["batch"] == audio.recommend(24.0, 5.0)
    r = c.get("/api/models/qwen3-8b/batch", params={"vram_gb": 24, "item_tokens": 1024, "n": 8})
    assert r.status_code == 200 and r.json()["recommendation"]["batch"] == kv.recommend(24.0, 1024)
