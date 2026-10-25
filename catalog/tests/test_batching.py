"""Batch profiles: the fit, recommendations, start limits, the vLLM log, the sweep, the API and the start env."""

from __future__ import annotations

import pytest
from catalog_helpers import step
from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.batching import Memory, Oom, Point, Profile, fit, kv_profile, lengths_for, mode, sweep, table
from tools.schema import load_platform

TRUE = (1.2, 0.004, 0.0003)  # base GiB, a per item-second, b per item-second²


def peak(n: int, length: float) -> float:
    base, a, b = TRUE
    return base + a * n * length + b * n * length * length


def points() -> list[Point]:
    return [Point(n, L, round(peak(n, L), 4), 0.1) for L in (2.0, 15.0, 60.0) for n in (1, 2, 4, 8, 16)]


def test_fit_recovers_the_coefficients():
    base, a, b = fit(points())
    assert base == pytest.approx(TRUE[0], abs=1e-3) and a == pytest.approx(TRUE[1], rel=0.02)
    assert b == pytest.approx(TRUE[2], rel=0.02)


def test_fit_keeps_coefficients_non_negative():
    flat = [Point(n, L, 2.0 - 0.0001 * n * L, 0.1) for L in (2.0, 20.0) for n in (1, 2, 4)]
    base, a, b = fit(flat)
    assert a >= 0 and b >= 0 and base == pytest.approx(sum(p.peak_gib for p in flat) / len(flat), abs=1e-6)


def test_recommendation_is_the_largest_batch_under_the_headroom():
    prof = Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2])
    n = prof.recommend(24.0, 20.0)
    assert prof.peak(n, 20.0) <= 0.9 * 24.0 < prof.peak(n + 1, 20.0)
    assert prof.recommend(1.0, 20.0) == 0  # the base alone is over 0.9 GiB
    assert prof.recommend(32.0, 0.5) == 256  # capped
    assert prof.need(n, 20.0) == pytest.approx(prof.peak(n, 20.0) / 0.9)


def test_start_env_limits_items_and_padded_length():
    prof = Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=60.0, points=points())
    env = prof.start_env(24.0)
    n, padded_ms = int(env["AISVC_MAX_BATCH"]), int(env["AISVC_MAX_BATCH_LEN"])
    assert n == prof.recommend(24.0, 2.0)
    longest_n = padded_ms // 60_000  # a batch of the longest clips under the padded limit fits
    assert longest_n >= 1 and prof.peak(longest_n, 60.0) <= 0.9 * 24.0 + 1e-6
    assert Profile(kind="kv", kv_tokens=1000, kv_gib=1.0, fixed_gib=5.0).start_env(24.0) == {}


VLLM_LOG = """
INFO 10-24 12:00:01 [gpu_worker.py:298] Available KV cache memory: 10.50 GiB
INFO 10-24 12:00:01 [kv_cache_utils.py:1087] GPU KV cache size: 76,448 tokens
INFO 10-24 12:00:01 [kv_cache_utils.py:1091] Maximum concurrency for 8,192 tokens per request: 9.33x
"""


def test_kv_profile_from_the_vllm_log():
    prof = kv_profile(VLLM_LOG, 29.6)
    assert prof.kv_tokens == 76448 and prof.kv_gib == 10.5 and prof.fixed_gib == pytest.approx(19.1)
    assert prof.max_model_len == 8192 and prof.max_concurrency == 9.33
    assert prof.recommend(29.6, 8192) == 9  # vLLM's 9.33x
    assert prof.recommend(24.0, 8192) < 9 and prof.recommend(19.0, 8192) == 0
    with pytest.raises(ValueError):
        kv_profile("no kv lines", 29.6)


def test_table_rows():
    prof = Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=20.0)
    t = table(prof, 29.76)
    assert t["budgets_gb"] == [16.0, 24.0, 32.0, 29.76] and [r["length"] for r in t["rows"]] == [5.0, 20.0]
    assert all(len(r["batch"]) == 4 for r in t["rows"])


def test_modes_and_lengths():
    p = load_platform()
    from tools import render

    assert mode(render.endpoint_entry(p, p.get("wavlm-sv"))) == "oip-batch"
    assert mode(render.endpoint_entry(p, p.get("faster-whisper-large-v3"))) == "openai-asr"
    assert mode({"contract": "openai", "api": "chat.completions"}) is None
    assert lengths_for(p.get("utmos")) == ([2.0, 15.0, 60.0], None)


class FakeMemory(Memory):
    """Peak from the true formula; out of memory past `limit` GiB."""

    def __init__(self, limit: float):
        self.limit = limit
        self.last: tuple[int, float] | None = None

    def measure(self, fn):
        fn()
        n, length = self.last
        if peak(n, length) > self.limit:
            raise Oom("CUDA out of memory")
        return round(peak(n, length), 4), "fake"


def test_sweep_stops_at_the_first_out_of_memory_point():
    mem = FakeMemory(limit=7.0)

    def send(n, length):
        mem.last = (n, length)

    pts, stops, how = sweep(send, mem, [2.0, 15.0, 60.0], budget_gib=None, max_n=64)
    assert how == "fake" and {p.length_s for p in pts} == {2.0, 15.0, 60.0}
    assert max(p.n for p in pts if p.length_s == 2.0) == 64 and stops[0]["stopped"] == "max batch"
    long = [p.n for p in pts if p.length_s == 60.0]
    assert long == [1, 2, 4] and "out of memory at 8" in stops[2]["stopped"]  # 8 x 60 s needs 11.7 GiB
    base, a, b = fit(pts)
    assert b == pytest.approx(TRUE[2], rel=0.05)


def test_memory_reads_device_and_torch_and_spots_a_backoff():
    class Http:
        def __init__(self):
            self.backoffs, self.calls = 0, []

        def post(self, url, **kw):
            self.calls.append(url)
            return R({"cuda": True, "reserved_gib": 1.0, "batch": {"m": {"backoffs": self.backoffs}}})

        def get(self, url, **kw):
            return R({"cuda": True, "peak_reserved_gib": 3.5, "batch": {"m": {"backoffs": self.backoffs}}})

    class R:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    used = iter([4.0, 4.5, 6.0, 5.0] + [5.0] * 100)
    http = Http()
    m = Memory("http://svc", http, lambda: (32.0, 32.0 - next(used)), baseline_gib=2.0)
    gib, how = m.measure(lambda: None)
    # torch: idle 4.0 - baseline 2.0 + (3.5 - 1.0) = 4.5; nvidia-smi max(...) - 2.0 >= 2.5
    assert gib >= 4.5 and http.calls == ["http://svc/platform/memory/reset"]

    def lowers():
        http.backoffs += 1

    with pytest.raises(Oom):
        m.measure(lowers)


def test_batch_api_without_and_with_a_profile(cat):
    from catalog.batching import Profile

    with TestClient(create_app(cat, start=False)) as c:
        r = c.get("/api/models/wavlm-sv/batch")
        assert r.status_code == 200 and r.json()["status"] == "none"
        assert c.get("/api/models/no-such/batch").status_code == 404
        r = c.post("/api/models/wavlm-sv/batch-profile")
        assert r.status_code == 202 and r.json()["measuring"] == "pending"

        cat.profiler._save("wavlm-sv", "bk1", "measured",
                           Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=20.0), None)
        v = cat.profiler.view("wavlm-sv", "bk1", vram_gb=24.0, item_s=20.0, item_tokens=None, n=8, full_gib=29.76)
        assert v["recommendation"]["batch"] == Profile(kind="audio", base=TRUE[0], a=TRUE[1],
                                                       b=TRUE[2]).recommend(24.0, 20.0)
        assert v["need"]["vram_gb"] == pytest.approx(peak(8, 20.0) / 0.9, abs=0.01)
        assert v["table"]["rows"][0]["length"] == 5.0
        assert cat.profiler.start_env("wavlm-sv", "bk1", 24.0)["AISVC_MAX_BATCH"]
        assert cat.profiler.start_env("wavlm-sv", "other-build", 24.0) == {}


def test_a_recommendation_needs_a_length_in_the_profiles_unit_and_range(cat):
    kv = Profile(kind="kv", kv_tokens=200_000, kv_gib=20.0, fixed_gib=5.0)
    cat.profiler._save("qwen3-8b", "bk-kv", "measured", kv, None)
    v = cat.profiler.view("qwen3-8b", "bk-kv", vram_gb=None, item_s=20.0, item_tokens=None, n=8, full_gib=29.76)
    assert "recommendation" not in v and "need" not in v and "item_tokens" in v["note"]  # not 20 tokens
    v = cat.profiler.view("qwen3-8b", "bk-kv", vram_gb=None, item_s=20.0, item_tokens=1024, n=None, full_gib=29.76)
    assert v["recommendation"]["batch"] == kv.recommend(29.76, 1024) and "note" not in v

    cat.profiler._save("wavlm-sv", "bk1", "measured",
                       Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=15.0), None)
    v = cat.profiler.view("wavlm-sv", "bk1", vram_gb=24.0, item_s=20.0, item_tokens=1024, n=8, full_gib=29.76)
    assert "recommendation" not in v and "need" not in v and "15 s" in v["note"]
    v = cat.profiler.view("wavlm-sv", "bk1", vram_gb=24.0, item_s=15.0, item_tokens=None, n=None, full_gib=29.76)
    assert v["recommendation"]["unit"] == "s" and "note" not in v


def test_a_start_passes_the_batch_limits_and_records_memory_before(cat, runtime):
    from tools.vram_check import BATCH_VARS

    runtime.gpu_total_gib, runtime.gpu_other_gib = 32.0, 2.0
    seen = []

    def batch_env(name, build_key, gib):
        seen.append((name, gib))
        return {"AISVC_MAX_BATCH": "12", "AISVC_MAX_BATCH_LEN": "240000", "OTHER": "x"}

    cat.scheduler.batch_env = batch_env
    cat.scheduler.request("manual", ["wavlm-sv"])
    step(cat)
    env = runtime.envs["wavlm-sv"]
    assert env[BATCH_VARS["AISVC_MAX_BATCH"]] == "12" and env[BATCH_VARS["AISVC_MAX_BATCH_LEN"]] == "240000"
    assert "OTHER" not in env and seen == [("wavlm-sv", pytest.approx(29.0, abs=0.02))]  # 32 - 2 - 1 GiB
    assert cat.scheduler.runs["wavlm-sv"].gpu_used_before == pytest.approx(2.0)


def test_just_batch_prints_the_recommendation_and_table(cat, monkeypatch, capsys):
    import aicore.catalog
    from tools import batch as cli

    cat.profiler._save("wavlm-sv", "bk1", "measured",
                       Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=20.0, method="torch"), None)
    monkeypatch.setattr(cat.builds, "current", lambda p, s, c: type("B", (), {"build_key": "bk1"})())
    with TestClient(create_app(cat, start=False)) as api:
        monkeypatch.setattr(cli, "Catalog", lambda timeout: aicore.catalog.Catalog("http://testserver", http=api))
        assert cli.main(["show", "wavlm-sv", "--vram", "24", "--item-s", "20", "--n", "8"]) == 0
    out = capsys.readouterr().out
    assert "profile: measured" in out and "24 GB, items of 20 s: batch" in out and "needs" in out and "(full)" in out


def test_sweep_stops_before_a_batch_predicted_over_the_budget():
    mem = FakeMemory(limit=1e9)  # no out of memory error ever (WSL2 spills to system RAM instead)

    def send(n, length):
        mem.last = (n, length)

    pts, stops, _ = sweep(send, mem, [15.0], budget_gib=8.0, max_n=128)
    assert all(p.peak_gib <= 0.95 * 8.0 for p in pts) and "would need about" in stops[0]["stopped"]
    assert max(p.n for p in pts) == 32  # 64 x 15 s would need about 9.4 GiB


def test_kv_profile_from_a_vllm_0_31_log():
    """v0.31 prints the KV cache size and the concurrency on one line (measured on qwen3-8b, 2026-10-25)."""
    text = ("INFO [gpu_worker.py:692] Available KV cache memory: 17.88 GiB\n"
            "INFO [kv_cache_utils.py:2464] GPU KV cache size: 130,208 tokens, Maximum concurrency for 32,768 tokens "
            "per request: 3.97x\n")
    prof = kv_profile(text, 28.66)
    assert prof.kv_tokens == 130208 and prof.max_model_len == 32768 and prof.max_concurrency == 3.97
    assert prof.recommend(28.66, 32768) == 3


def test_no_request_carries_more_than_the_audio_cap():
    from catalog.batching import MAX_REQUEST_S

    prof = Profile(kind="audio", base=1.0, a=0.0001, b=0.0, max_item_s=180.0)
    assert prof.recommend(29.6, 180.0) == int(MAX_REQUEST_S // 180)
    mem = FakeMemory(limit=1e9)

    def send(n, length):
        mem.last = (n, length)

    pts, stops, _ = sweep(send, mem, [180.0], budget_gib=1e6, max_n=128)
    assert max(p.n for p in pts) == 32 and "over 6000 s" in stops[0]["stopped"]
