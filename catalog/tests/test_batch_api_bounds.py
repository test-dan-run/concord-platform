"""Batch API inputs that would break a start or the response: an audio limit that rounds to 0 ms (the image
refuses AISVC_MAX_BATCH_LEN=0), and query values that overflow the recommendation (500 before)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_batching import TRUE, points

from catalog.app import create_app
from catalog.batching import Profile


def test_an_audio_limit_under_a_millisecond_is_refused(cat, monkeypatch):
    with pytest.raises(ValueError):
        cat.profiler.set_override("wavlm-sv", None, 0.0004)  # start_env would give AISVC_MAX_BATCH_LEN "0"
    assert cat.profiler.override("wavlm-sv") is None
    cat.profiler.set_override("wavlm-sv", None, 0.001)
    assert cat.profiler.start_env("wavlm-sv", "unmeasured", 24.0) == {"AISVC_MAX_BATCH_LEN": "1"}
    with TestClient(create_app(cat, start=False)) as c:
        assert c.put("/api/models/wavlm-sv/batch-override", json={"max_len_s": 0.0004}).status_code == 422


AUDIO = [{"vram_gb": "inf", "item_s": 5}, {"vram_gb": "nan", "item_s": 5}, {"vram_gb": 1e308, "item_s": 5},
         {"vram_gb": 24, "item_s": "inf"}, {"vram_gb": 24, "item_s": "nan"}, {"vram_gb": 24, "item_s": 1e-310},
         {"vram_gb": 24, "item_s": 1e300}, {"vram_gb": 24, "item_s": 5, "n": 10**400}]
KV = [{"item_tokens": 10**400}, {"item_tokens": 10**300}, {"vram_gb": 24, "item_tokens": 1000, "n": 10**300}]


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
