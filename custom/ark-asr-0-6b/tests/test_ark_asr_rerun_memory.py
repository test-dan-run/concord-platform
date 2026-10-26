"""ark-asr-0-6b: when a shared batch fails, its memory is released before the files run again one by one, and the
CUDA cache is emptied in between."""

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
asr_mod = load_server(SERVICE_DIR, "ark")


class Activations:  # stands in for the tensors a failing generate() call holds
    pass


class TrackingArk(asr_mod.ArkAsr):
    """Fails any batch that holds a 3 s chunk, and records whether a failed call's locals are still alive."""

    def __init__(self, error: str):
        super().__init__("/nonexistent")
        self.error = error
        self.failed: list[weakref.ref] = []
        self.alive: list[bool] = []  # one entry per call: was an earlier failed call's memory still referenced?

    def load(self):
        self.model = object()

    def failed_alive(self) -> bool:
        gc.collect()
        return any(r() is not None for r in self.failed)

    def transcribe_batch(self, pieces):
        self.alive.append(self.failed_alive())
        activations = Activations()  # a frame local, like the tensors in a real generate()
        if any(len(x) == 48000 for x in pieces):
            self.failed.append(weakref.ref(activations))
            raise RuntimeError(self.error)  # a fresh error each call, as torch raises
        return [f"chunk{len(x)}" for x in pieces]


def build_files(monkeypatch, model):
    """The files() function the app hands to its requests batcher, and the free_cuda_cache calls it makes."""
    monkeypatch.setitem(sys.modules, "torch", None)  # is_oom then goes by the message alone
    captured, freed = {}, []
    real = app_mod.DynamicBatcher

    def spy(fn, **kw):
        captured["files"] = fn
        return real(fn, **kw)

    monkeypatch.setattr(app_mod, "DynamicBatcher", spy)
    monkeypatch.setattr(app_mod, "free_cuda_cache", lambda: freed.append(model.failed_alive()), raising=False)
    app_mod.build_app(model)
    return captured["files"], freed


@pytest.mark.parametrize("error", ["bad chunk", "CUDA out of memory"])
def test_failed_batch_is_released_before_files_run_again(monkeypatch, error):
    model = TrackingArk(error)
    files, freed = build_files(monkeypatch, model)
    group = [[np.zeros(16000, np.float32)], [np.zeros(48000, np.float32)], [np.zeros(32000, np.float32)]]
    out = files(group)
    assert out[0] == "chunk16000" and out[2] == "chunk32000"
    assert model.alive == [False] * len(model.alive)
    assert isinstance(out[1], RuntimeError) and str(out[1]) == error and out[1].__traceback__ is None
    assert freed == [False]  # once, before the reruns, with the failed batch already released


@pytest.mark.parametrize("error", ["bad chunk", "CUDA out of memory"])
def test_one_file_group_returns_its_error_without_a_rerun(monkeypatch, error):
    model = TrackingArk(error)
    files, freed = build_files(monkeypatch, model)
    out = files([[np.zeros(48000, np.float32)]])
    assert len(out) == 1 and isinstance(out[0], RuntimeError)
    assert len(model.alive) == 1 and freed == []
    assert not model.failed_alive() and out[0].__traceback__ is None
