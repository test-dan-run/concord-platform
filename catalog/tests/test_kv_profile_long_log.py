"""The vLLM batch profile reads the KV cache lines from the whole container log, and refuses a service whose
start the catalog did not size."""

import threading
import time

import pytest
from catalog_helpers import step

from catalog.runtime import FakeRuntime

KV_OLD = ("INFO [gpu_worker.py:692] Available KV cache memory: 9.00 GiB\n"
          "INFO [kv_cache_utils.py:2464] GPU KV cache size: 65,000 tokens, Maximum concurrency for 32,768 tokens "
          "per request: 1.98x\n")
KV = ("INFO [gpu_worker.py:692] Available KV cache memory: 17.88 GiB\n"
      "INFO [kv_cache_utils.py:2464] GPU KV cache size: 130,208 tokens, Maximum concurrency for 32,768 tokens "
      "per request: 3.97x\n")
ACCESS = 'INFO:     172.18.0.1:40000 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'


class TailRuntime(FakeRuntime):
    """Like `docker compose logs --tail N`: only the last `tail` lines."""

    def logs(self, p, name, tail):
        return "".join(super().logs(p, name, tail).splitlines(keepends=True)[-tail:])


@pytest.fixture
def runtime() -> TailRuntime:
    return TailRuntime()


def profile(cat, model: str) -> None:
    """Run the profiler on its own thread while this one drives the scheduler, so ticks and starts never race."""
    t = threading.Thread(target=cat.profiler.run, args=(model,), daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while t.is_alive():
        assert time.monotonic() < deadline, "the profile did not finish"
        step(cat, 1)
        t.join(0.05)


def test_the_kv_lines_are_found_after_many_requests_and_the_last_start_wins(cat, runtime):
    # a restart: the first start's lines, its requests, then the current start's lines and 1000 requests
    runtime.log_text["qwen3-8b"] = KV_OLD + ACCESS * 500 + KV + ACCESS * 1000
    profile(cat, "qwen3-8b")
    row = cat.db.one("SELECT status, error FROM batch_profile WHERE model_id = 'qwen3-8b'")
    assert (row["status"], row["error"]) == ("measured", None)
    prof = cat.profiler.stored(cat.scheduler.runs["qwen3-8b"].build.build_key)["profile"]
    assert prof["kv_tokens"] == 130208 and prof["kv_gib"] == 17.88 and prof["max_concurrency"] == 3.97


def test_a_service_started_outside_the_catalog_is_not_measured(cat, runtime):
    runtime.log_text["qwen3-8b"] = KV
    runtime.running["qwen3-8b"] = 0.0  # up before the catalog: it never gave this start its memory
    cat.scheduler.tick()
    assert cat.scheduler.runs["qwen3-8b"].running and cat.scheduler.runs["qwen3-8b"].given is None
    profile(cat, "qwen3-8b")
    assert cat.scheduler.runs["qwen3-8b"].given is None
    row = cat.db.one("SELECT status, error FROM batch_profile WHERE model_id = 'qwen3-8b'")
    assert row["status"] == "failed"
    assert row["error"] == ("started outside the catalog (or before the catalog restarted), so the memory it was "
                            "given is unknown: restart it from the catalog to measure")
