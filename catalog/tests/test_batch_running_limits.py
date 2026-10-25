"""The Batching view gives the limits a start with the running service's memory gets now (running.limits), so the
panel can tell whether what the service started with is still current. The top-level `limits` are at the full
budget, which a start almost never gets (it gets the memory free less 1 GiB)."""

from __future__ import annotations

from catalog_helpers import step
from fastapi.testclient import TestClient
from test_batching import TRUE, points

from catalog.app import create_app
from catalog.batching import Profile
from tools.vram_check import VRAM_VAR


def test_running_limits_are_the_limits_its_start_got(cat, runtime, monkeypatch):
    # 29.95 GiB free of 32: the start gets 0.904 of the device, 28.928 GiB, floored to 28.92 (rounding gives 28.93)
    runtime.gpu_total_gib, runtime.gpu_other_gib = 32.0, 2.05
    prof = Profile(kind="audio", base=2.04, a=0.0186, b=0.0, max_item_s=20.0, points=points())
    cat.profiler._save("wavlm-sv", "bk1", "measured", prof, None)
    monkeypatch.setattr(cat.builds, "current", lambda p, s, c: type("B", (), {"build_key": "bk1"})())
    cat.scheduler.request("manual", ["wavlm-sv"])
    step(cat)
    assert cat.scheduler.runs["wavlm-sv"].running
    with TestClient(create_app(cat, start=False)) as c:
        v = c.get("/api/models/wavlm-sv/batch").json()
    r = v["running"]
    assert r["vram_gb"] == float(runtime.envs["wavlm-sv"][VRAM_VAR]) == 28.92
    assert {k: r["limits"][k] for k in ("max_batch", "max_len_s")} == r["started_with"]
    assert r["started_with"] != {k: v["limits"][k] for k in ("max_batch", "max_len_s")}  # those are at 29.76 GiB

    cat.profiler.set_override("wavlm-sv", 6, None)  # after the start: it applies the next time
    with TestClient(create_app(cat, start=False)) as c:
        r = c.get("/api/models/wavlm-sv/batch").json()["running"]
    assert r["limits"]["max_batch"] == 6 and r["started_with"]["max_batch"] != 6


def test_no_running_limits_for_a_start_outside_the_catalog_or_a_vllm_service(cat, monkeypatch):
    cat.profiler._save("wavlm-sv", "bk1", "measured", Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2]), None)
    monkeypatch.setattr(cat.builds, "current", lambda p, s, c: type("B", (), {"build_key": "bk1"})())
    cat.scheduler.platform()
    run = cat.scheduler.runs["wavlm-sv"]
    run.running, run.given, run.batch = True, None, None  # started outside the catalog
    vllm = cat.scheduler.runs["bge-m3"]
    vllm.running, vllm.given, vllm.batch = True, 0.5, None
    with TestClient(create_app(cat, start=False)) as c:
        assert "limits" not in c.get("/api/models/wavlm-sv/batch").json()["running"]
        assert "limits" not in c.get("/api/models/bge-m3/batch").json()["running"]
