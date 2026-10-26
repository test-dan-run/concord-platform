"""aisvc.batch: micro-batching, out of memory backoff, dynamic batching, and GET /platform/memory."""

import sys
import threading
import types

import numpy as np
import pytest
from aisvc import DynamicBatcher, Limits, MicroBatcher, create_oip_app, create_openai_app
from aisvc.batch import pad, unpad
from fastapi.testclient import TestClient


class FakeOOM(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


@pytest.fixture
def fake_torch(monkeypatch):
    """A torch module in sys.modules with just what aisvc touches, so tests need no real torch."""
    cuda = types.SimpleNamespace(OutOfMemoryError=FakeOOM, is_available=lambda: True, emptied=0, peak=5 * 1024**3)

    def empty_cache():
        cuda.emptied += 1

    def reset_peak_memory_stats(device):
        cuda.peak = 0

    cuda.empty_cache = empty_cache
    cuda.reset_peak_memory_stats = reset_peak_memory_stats
    cuda.mem_get_info = lambda device: (20 * 1024**3, 32 * 1024**3)
    cuda.max_memory_allocated = lambda device: cuda.peak
    cuda.max_memory_reserved = lambda device: 6 * 1024**3
    cuda.memory_allocated = lambda device: 2 * 1024**3
    cuda.memory_reserved = lambda device: 3 * 1024**3
    cuda.get_device_name = lambda device: "Fake GPU"
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def recorder(fail=None):
    """A batch function that returns each item doubled and records the chunks it was given."""
    chunks = []

    def fn(batch):
        if fail:
            fail(batch)
        chunks.append(list(batch))
        return [x * 2 for x in batch]

    return fn, chunks


def test_results_keep_input_order_after_the_length_sort():
    fn, chunks = recorder()
    items = ["ccc", "a", "eeeee", "bb", "dddd"]
    run = MicroBatcher(fn, limits=Limits(max_items=2))
    assert run(items) == [x * 2 for x in items]
    assert chunks == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]  # sorted by length, so chunks pad little


def test_chunks_respect_max_padded_length():
    fn, chunks = recorder()
    run = MicroBatcher(fn, length=lambda x: x, limits=Limits(max_items=100, max_len=10))
    assert run([5, 1, 50, 4, 2, 3]) == [10, 2, 100, 8, 4, 6]
    # 3 x 3 = 9 fits; 4 x 4 = 16 does not; 2 x 5 = 10 fits; 50 is over the cap but a single item still runs
    assert chunks == [[1, 2, 3], [4, 5], [50]]


def test_limits_come_from_env(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "16")
    monkeypatch.setenv("AISVC_MAX_BATCH_LEN", "480000")
    lim = Limits.from_env(max_items=4)
    assert (lim.max_items, lim.max_len) == (16, 480000)
    monkeypatch.delenv("AISVC_MAX_BATCH")
    monkeypatch.delenv("AISVC_MAX_BATCH_LEN")
    assert (Limits.from_env().max_items, Limits.from_env().max_len) == (8, None)
    assert Limits.from_env(max_items=2, max_len=99).max_len == 99  # an image's own defaults
    monkeypatch.setenv("AISVC_MAX_BATCH", "0")
    with pytest.raises(ValueError, match="AISVC_MAX_BATCH must be at least 1"):
        Limits.from_env()


def test_out_of_memory_halves_the_chunk_and_remembers(fake_torch):
    def fail(batch):
        if len(batch) * max(batch) > 8:
            raise FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")

    fn, chunks = recorder(fail)
    lim = Limits(max_items=8)
    run = MicroBatcher(fn, length=lambda x: x, limits=lim)
    assert run([4, 4, 4, 4]) == [8, 8, 8, 8]
    assert chunks == [[4, 4], [4, 4]] and fake_torch.cuda.emptied == 1
    assert lim.stats() == {"max_items": 8, "max_len": 8, "configured_max_items": 8, "configured_max_len": None,
                           "backoffs": 1, "lowered": True}
    chunks.clear()
    assert run([4, 4, 4]) == [8, 8, 8]  # the lower limit is kept: no second out of memory error
    assert chunks == [[4, 4], [4]] and lim.backoffs == 1


def test_out_of_memory_backoff_needs_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # a CTranslate2 or ONNX image

    def fail(batch):
        if len(batch) > 1:
            raise RuntimeError("CUDA failed with error out of memory")

    fn, chunks = recorder(fail)
    assert MicroBatcher(fn, length=lambda x: 1, limits=Limits(max_items=4))([1, 2, 3]) == [2, 4, 6]
    assert chunks == [[1], [2], [3]]


def test_a_single_item_that_runs_out_of_memory_raises(fake_torch):
    def fail(batch):
        raise FakeOOM("CUDA out of memory")

    fn, _ = recorder(fail)
    with pytest.raises(FakeOOM):
        MicroBatcher(fn, length=lambda x: x, limits=Limits())([3, 3])  # splits to single items, which still fail
    assert fake_torch.cuda.emptied >= 2


def test_other_errors_are_not_retried():
    calls = []

    def fn(batch):
        calls.append(batch)
        raise ValueError("bad clip")

    with pytest.raises(ValueError, match="bad clip"):
        MicroBatcher(fn, limits=Limits())(["a", "b"])
    assert len(calls) == 1


def test_wrong_result_count_is_an_error():
    with pytest.raises(RuntimeError, match="returned 1 results for 2 items"):
        MicroBatcher(lambda batch: [0], limits=Limits())(["a", "b"])


def run_together(batcher, items):
    """Call `batcher` from one thread per item at the same moment; returns results (or errors) by item."""
    out, start = {}, threading.Barrier(len(items))

    def call(item):
        start.wait()
        try:
            out[item] = batcher(item)
        except Exception as exc:
            out[item] = exc

    threads = [threading.Thread(target=call, args=(it,)) for it in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return out


def test_dynamic_batcher_groups_concurrent_calls_by_key():
    fn, chunks = recorder()
    batcher = DynamicBatcher(fn, key=lambda item: item[:2], wait_ms=300, limits=Limits(max_items=16))
    items = ["en1", "en2", "en3", "en4", "fr1", "fr2"]
    assert run_together(batcher, items) == {it: it * 2 for it in items}  # each caller gets its own result
    assert sorted(sorted(c) for c in chunks) == [["en1", "en2", "en3", "en4"], ["fr1", "fr2"]]


def test_dynamic_batcher_gives_a_batch_error_to_all_its_callers():
    def fail(batch):
        if "en-bad" in batch:
            raise ValueError("could not decode")

    fn, _ = recorder(fail)
    batcher = DynamicBatcher(fn, key=lambda item: item[:2], wait_ms=300, limits=Limits())
    out = run_together(batcher, ["en-ok", "en-bad", "fr-ok"])
    assert isinstance(out["en-ok"], ValueError) and isinstance(out["en-bad"], ValueError)
    assert out["fr-ok"] == "fr-ok" * 2


def test_pad_and_unpad():
    padded, lengths = pad([np.ones(3), np.ones(1)])
    assert padded.shape == (2, 3) and padded.dtype == np.float32 and lengths.tolist() == [3, 1]
    assert padded[1].tolist() == [1, 0, 0]
    assert [r.tolist() for r in unpad(padded, lengths)] == [[1, 1, 1], [1]]


APPS = {
    "oip": lambda: create_oip_app("m", {"inputs": [{"name": "X", "datatype": "FP32", "shape": [-1]}],
                                        "outputs": [{"name": "Y", "datatype": "FP32", "shape": [-1]}]}, lambda i, p: {"Y": i["X"]}),
    "openai": lambda: create_openai_app("m", "audio.transcriptions", lambda req: "hi"),
}


def test_memory_route_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    MicroBatcher(lambda batch: batch, limits=Limits(max_items=4, max_len=1000), name="mem-test")
    with TestClient(APPS["openai"]()) as c:
        body = c.get("/platform/memory").json()
        assert body["torch"] is False and body["cuda"] is False and body["reset"] is False
        assert body["batch"]["mem-test"]["max_items"] == 4 and body["batch"]["mem-test"]["max_len"] == 1000
        assert c.post("/platform/memory/reset").json()["torch"] is False


def test_memory_route_with_cuda_reports_and_resets_the_peak(fake_torch, monkeypatch):
    monkeypatch.setenv("AISVC_VRAM_GB", "6")
    with TestClient(APPS["oip"]()) as c:
        body = c.get("/platform/memory").json()
        assert body["torch"] is True and body["cuda"] is True and body["device_name"] == "Fake GPU"
        assert body["peak_allocated_bytes"] == 5 * 1024**3 and body["peak_allocated_gib"] == 5.0
        assert body["device_total_gib"] == 32.0 and body["device_free_gib"] == 20.0 and body["reserved_gib"] == 3.0
        assert body["vram_gb_limit"] == "6"
        reset = c.post("/platform/memory/reset").json()
        assert reset["reset"] is True and reset["peak_allocated_gib"] == 5.0  # read, then cleared
        assert c.get("/platform/memory").json()["peak_allocated_bytes"] == 0


def test_batch_limits_route_lifts_and_restores():
    from fastapi.testclient import TestClient

    from aisvc import batch
    from aisvc._core import create_base_app

    mb = batch.MicroBatcher(lambda xs: xs, length=lambda x: 1, limits=batch.Limits(4, 100), name="route-test")
    with TestClient(create_base_app("t", title="t", load=None, background=False)) as c:
        r = c.post("/platform/batch-limits", json={"max_items": 512, "max_len": None})
        assert r.status_code == 200 and r.json()["route-test"]["max_items"] == 512
        assert mb.limits.max_len is None and mb(list(range(300))) == list(range(300))
        assert c.post("/platform/batch-limits", json={"max_items": 0}).status_code == 422
        assert c.post("/platform/batch-limits", json={"max_len": 1000}).status_code == 200
        assert (mb.limits.max_items, mb.limits.max_len) == (512, 1000)  # no max_items: it keeps its value
        assert c.post("/platform/batch-limits", json={"max_len": None}).status_code == 200
        assert (mb.limits.max_items, mb.limits.max_len) == (512, None)
        r = c.post("/platform/batch-limits", json={"restore": True})
        assert r.json()["route-test"] == {**r.json()["route-test"], "max_items": 4, "max_len": 100, "backoffs": 0}
    batch._LIMITS.pop("route-test", None)


def test_audio_ms():
    from aisvc.batch import audio_ms

    assert audio_ms([0.0] * 16000, 16000) == 1000 and audio_ms([0.0] * 24000, 48000) == 500
