"""A swap is counted once per eviction: an eviction tried again after its stop failed is the same swap, but a
failed spin-down stop earlier does not hide the swap of a later eviction."""

from catalog_helpers import drain, step

M = "whisper-large-v3"
OTHER = "qwen3-32b-awq"


def test_a_swap_after_a_failed_spin_down_stop_is_counted(cat, runtime):
    s = cat.scheduler
    cat.settings.release_linger_s = 0.0
    a = s.request("bench:a", [M])
    step(cat)
    real = runtime.stop

    def failing(p, names):
        raise RuntimeError("docker compose stop exited 1")

    runtime.stop = failing
    s.release(a.id)
    s.tick()  # the spin-down queues a stop
    drain(cat)  # which fails once
    assert s.runs[M].running and s.runs[M].stop_failures == 1
    runtime.stop = real
    before = s.swaps
    b = s.request("bench:b", [OTHER])  # evicts M, then starts OTHER
    step(cat)
    assert s.get(b.id).state == "active" and sorted(runtime.running) == [OTHER]
    assert s.swaps - before == 1
