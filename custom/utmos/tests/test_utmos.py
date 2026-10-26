"""utmos: oip-v2 conformance with a fake model, batch handling, and the real model when weights and torch exist."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
mos_mod = load_server(SERVICE_DIR, "utmos")
META = service_meta("utmos")


class FakeUtmos(mos_mod.Utmos):
    """Score grows with RMS, so a louder clip scores higher; records what it was given."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.seen = []

    def load(self):
        self.model = object()

    def score_batch(self, clips):
        self.seen.extend((len(a), sr) for a, sr in clips)
        return [2.0 + float(np.sqrt(np.mean(np.asarray(a, np.float32) ** 2))) for a, _ in clips]


@pytest.fixture(scope="module")
def model():
    return FakeUtmos()


@pytest.fixture(scope="module")
def client(model):
    with running(app_mod.build_app(model)) as c:
        yield c


def infer(client, audio, lengths, sr=16000):
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [len(lengths)], "data": list(lengths)},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    return client.post("/v2/models/utmos/infer", json=body)


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_batch_uses_lengths(client, model):
    a = np.zeros((2, 32000), np.float32)
    a[0, :16000] = 0.5
    a[1, :] = 0.5
    model.seen.clear()
    outs = {o["name"]: o for o in infer(client, a, [16000, 32000]).json()["outputs"]}
    assert outs["MOS"]["shape"] == [2]
    assert outs["MOS"]["data"][0] == pytest.approx(outs["MOS"]["data"][1], abs=1e-5)  # padding is ignored
    assert model.seen == [(16000, 16000), (32000, 16000)]


@pytest.mark.parametrize("lengths, sr", [([9], 16000), ([0], 16000), ([4], 4000)])
def test_bad_input_is_400(client, lengths, sr):
    assert infer(client, np.zeros((1, 4), np.float32), lengths, sr).status_code == 400


def test_rows_and_lengths_must_agree(client):
    assert infer(client, np.zeros((2, 4), np.float32), [4]).status_code == 400


WEIGHTS = Path(os.environ.get("UTMOS_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/utmos/utmos22-strong-eecd3e6"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
def test_real_model_prefers_clean_over_noisy():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import wave

    m = mos_mod.Utmos(WEIGHTS)
    m.load()
    # Real speech: UTMOS scores a synthetic tone about the same with or without noise (1.31 against 1.36).
    with wave.open(str(SERVICE_DIR.parents[1] / "catalog/fixtures/check_clip.wav")) as w:
        clean = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    noisy = clean + np.random.default_rng(0).normal(0, 0.05, len(clean)).astype(np.float32)
    assert m.score(clean, 16000) > m.score(noisy, 16000)
    assert m.score(clean, 16000) == pytest.approx(m.score(clean, 16000))  # deterministic


def _speechlike(seconds: float, f0: float, seed: int) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    a = 0.3 * np.sin(2 * np.pi * f0 * t) * (1 + np.sin(2 * np.pi * 3.7 * t))
    return (a + np.random.default_rng(seed).normal(0, 0.02, len(t))).astype(np.float32)


MIXED = [_speechlike(s, f, k) for k, (s, f) in enumerate([(1.3, 140), (4.0, 210), (7.5, 120), (25.0, 180), (0.3, 160)])]


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no checkpoint in {WEIGHTS}")
def test_real_model_batch_matches_one_by_one():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    m = mos_mod.Utmos(WEIGHTS)
    m.load()
    clips = [(a, 16000) for a in MIXED] + [(MIXED[1], 24000)]
    batched = m.score_batch(clips)
    for c, b in zip(clips, batched, strict=True):
        assert abs(m.score_batch([c])[0] - b) < 0.02
