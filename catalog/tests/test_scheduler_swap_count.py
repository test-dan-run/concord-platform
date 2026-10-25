"""The swap count when GPU models were started outside the catalog: one per real swap, also after an eviction
that swapped nothing, an abandoned eviction, or two models evicted at once."""
import time

from catalog_helpers import drain, step

M = "whisper-large-v3"
OTHER = "qwen3-32b-awq"
X = "qwen2-5-14b-instruct-awq"


def _failing(p, names):
    raise RuntimeError("docker compose stop exited 1")


def test_two_gpu_models_up_an_uncounted_eviction_hides_a_later_real_swap(cat, runtime):
    """M and OTHER both run (started outside the catalog). A lease for M (already up) evicts OTHER: no swap,
    rightly. That stop fails, so OTHER keeps evicting=True. Later a request for X stops M and OTHER and starts X:
    a real swap, which must count once."""
    s = cat.scheduler
    runtime.running.update({M: 0.0, OTHER: 0.0})
    s.tick()
    for n in (M, OTHER):
        s.runs[n].ready_at = time.time() - 10_000  # settled
    real = runtime.stop
    runtime.stop = _failing
    before = s.swaps
    a = s.request("bench:a", [M])
    s.tick()
    drain(cat)
    s.tick()
    assert s.get(a.id).state == "active"
    assert s.runs[OTHER].running and s.swaps == before  # no swap yet: M was already up
    runtime.stop = real
    s.release(a.id)
    b = s.request("bench:b", [X])
    step(cat, 3)
    assert s.get(b.id).state == "active" and sorted(runtime.running) == [X]
    assert s.swaps - before == 1, f"real swap M/OTHER -> {X} counted {s.swaps - before} times"


def test_abandoned_eviction_then_real_swap_total_matches(cat, runtime):
    """B evicts M, the stop keeps failing, B ends. M (pinned) stays up with evicting=True. C then swaps M for X.
    The total equals the one real swap (B's count was made at its grant)."""
    s = cat.scheduler
    runtime.running.update({M: 0.0})
    s.tick()
    s.runs[M].ready_at = time.time() - 10_000
    assert s.runs[M].pinned
    real = runtime.stop
    runtime.stop = _failing
    before = s.swaps
    b = s.request("bench:b", [OTHER])
    for _ in range(3):
        s.tick()
        drain(cat)
    s.release(b.id)
    s.tick()
    assert s.runs[M].running and s.runs[M].evicting
    runtime.stop = real
    c = s.request("bench:c", [X])
    step(cat, 3)
    assert s.get(c.id).state == "active" and sorted(runtime.running) == [X]
    assert s.swaps - before == 1


def test_two_evicted_at_once_one_stop_fails_counts_once(cat, runtime):
    s = cat.scheduler
    runtime.running.update({M: 0.0, OTHER: 0.0})
    s.tick()
    for n in (M, OTHER):
        s.runs[n].ready_at = time.time() - 10_000
    real = runtime.stop

    def half(p, names):
        if names == [OTHER]:
            raise RuntimeError("docker compose stop exited 1")
        return real(p, names)

    runtime.stop = half
    before = s.swaps
    b = s.request("bench:b", [X])
    for _ in range(4):
        s.tick()
        drain(cat)
    assert s.swaps - before == 1 and X not in runtime.running
    runtime.stop = real
    step(cat, 3)
    assert s.get(b.id).state == "active" and sorted(runtime.running) == [X] and s.swaps - before == 1
