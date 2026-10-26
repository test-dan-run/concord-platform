"""An eviction whose stop fails: the next GPU model waits until the evicted one is down, and the swap counts once."""

from catalog_helpers import drain, step

M = "whisper-large-v3"
OTHER = "qwen3-32b-awq"


def evict_with_a_failing_stop(cat, runtime):
    """M ran under a lease that ended; its stop fails from now on. Returns the real stop."""
    s = cat.scheduler
    a = s.request("bench:a", [M])
    step(cat)
    s.release(a.id)
    assert s.runs[M].running
    real = runtime.stop

    def stop(p, names):
        raise RuntimeError("docker compose stop exited 1")

    runtime.stop = stop
    return real


def test_an_eviction_whose_stop_keeps_failing_counts_one_swap(cat, runtime):
    s = cat.scheduler
    cat.settings.gpu_free_wait_s = 0.0
    runtime.gpu_total_gib = 32.0
    s.tick()
    real = evict_with_a_failing_stop(cat, runtime)
    before = s.swaps
    b = s.request("bench:b", [OTHER])
    for _ in range(6):
        s.tick()
        drain(cat)
    assert s.swaps - before == 1 and OTHER not in runtime.started
    runtime.stop = real
    step(cat)
    assert s.get(b.id).state == "active" and sorted(runtime.running) == [OTHER] and s.swaps - before == 1


def test_a_gpu_model_waits_for_an_evicted_model_whose_stop_failed(cat, runtime):
    s = cat.scheduler
    assert runtime.gpu_total_gib is None  # no GPU reading: memory cannot tell that M is still loaded
    real = evict_with_a_failing_stop(cat, runtime)
    b = s.request("bench:b", [OTHER])
    drain(cat)
    s.tick()
    r = s.get(b.id)
    assert r.state == "queued" and r.waiting_for == f"waiting for {M} to stop"
    assert sorted(runtime.running) == [M] and OTHER not in runtime.started
    step(cat)
    assert sorted(runtime.running) == [M] and OTHER not in runtime.started
    runtime.stop = real
    step(cat)
    assert s.get(b.id).state == "active" and sorted(runtime.running) == [OTHER]
