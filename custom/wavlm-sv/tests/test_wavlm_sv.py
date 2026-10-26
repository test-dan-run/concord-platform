"""wavlm-sv: oip-v2 conformance with a fake model, input preparation, and the real model when weights exist."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
emb_mod = load_server(SERVICE_DIR, "embedder")
META = service_meta("wavlm-sv")
DIM = 256


class FakeEmbedder(emb_mod.WavLMEmbedder):
    """A unit vector from the clip's RMS and length, after the real input preparation."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.seen = []

    def load(self):
        self.model = object()

    def embed_batch(self, clips):
        out = []
        for x in clips:
            self.seen.append(len(x))
            v = np.zeros(DIM, np.float32)
            v[0], v[1] = 1.0, float(np.sqrt(np.mean(x ** 2)))
            out.append(v / np.linalg.norm(v))
        return out


@pytest.fixture(scope="module")
def model():
    return FakeEmbedder()


@pytest.fixture(scope="module")
def client(model):
    with running(app_mod.build_app(model)) as c:
        yield c


def infer(client, audio, lengths, sr=16000):
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [len(lengths)], "data": list(lengths)},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    return client.post(f"/v2/models/{META['name']}/infer", json=body)


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def test_batch_returns_one_unit_row_per_clip(client):
    a = np.zeros((2, 16000), np.float32)
    a[0, :8000] = 0.5
    a[1, :] = 0.5
    out = {o["name"]: o for o in infer(client, a, [8000, 16000]).json()["outputs"]}["EMBEDDINGS"]
    assert out["shape"] == [2, DIM]
    e = np.asarray(out["data"], np.float32).reshape(out["shape"])
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    assert np.allclose(e[0], e[1], atol=1e-5)  # padding is ignored


def test_prepare_repeats_short_clips_and_resamples():
    x = emb_mod.prepare(np.arange(100, dtype=np.float32), 16000)
    assert len(x) == int(emb_mod.MIN_S * emb_mod.SR) and x[100] == 0 and x[101] == 1
    assert len(emb_mod.prepare(np.zeros(48000, np.float32), 48000)) == 16000
    assert len(emb_mod.prepare(np.zeros(16000 * 70, np.float32), 16000)) == int(emb_mod.MAX_S * 16000)


@pytest.mark.parametrize("lengths, sr", [([9], 16000), ([0], 16000), ([4], 4000)])
def test_bad_input_is_400(client, lengths, sr):
    assert infer(client, np.zeros((1, 4), np.float32), lengths, sr).status_code == 400


WEIGHTS = Path(os.environ.get("WAVLM_SV_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/wavlm-sv/large-ecapa-13ad4fc"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no checkpoint in {WEIGHTS}")
def test_real_model_embeds_unit_vectors():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    m = emb_mod.WavLMEmbedder(WEIGHTS)
    m.load()
    t = np.arange(16000 * 2) / 16000
    a = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    e1, e2 = m.embed(a, 16000), m.embed(a * 0.5, 16000)
    assert e1.shape == (DIM,) and abs(float(np.linalg.norm(e1)) - 1) < 1e-4
    assert float(e1 @ e2) > 0.95  # gain does not change the speaker


def _speechlike(seconds: float, f0: float, seed: int) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    a = 0.3 * np.sin(2 * np.pi * f0 * t) * (1 + np.sin(2 * np.pi * 3.7 * t))
    return (a + np.random.default_rng(seed).normal(0, 0.02, len(t))).astype(np.float32)


MIXED = [_speechlike(s, f, k) for k, (s, f) in enumerate([(1.3, 140), (4.0, 210), (7.5, 120), (25.0, 180), (0.3, 160)])]


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no checkpoint in {WEIGHTS}")
def test_real_model_batch_matches_one_by_one():
    """Padding (and, for redimnet2, grouping by length) must not change an embedding."""
    pytest.importorskip("torch")
    m = emb_mod.WavLMEmbedder(WEIGHTS)
    m.load()
    clips = [emb_mod.prepare(a, 16000) for a in MIXED + MIXED[:2]]
    batched = m.embed_batch(clips)
    for c, b in zip(clips, batched, strict=True):
        one = m.embed_batch([c])[0]
        assert float(one @ b) > 0.999, float(one @ b)


def test_rows_run_in_batches_within_the_limit(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    model = FakeEmbedder()
    sizes = []
    inner = model.embed_batch
    model.embed_batch = lambda clips: (sizes.append(len(clips)), inner(clips))[1]
    with running(app_mod.build_app(model)) as c:
        audio = np.stack([np.resize(a, 16000 * 3) for a in MIXED])
        r = infer(c, audio, [16000, 32000, 48000, 8000, 24000])
    assert r.status_code == 200, r.text
    emb = np.asarray(r.json()["outputs"][0]["data"], np.float32).reshape(5, DIM)
    expected = np.stack(model.embed_batch([emb_mod.prepare(audio[i, :n], 16000)
                                           for i, n in enumerate([16000, 32000, 48000, 8000, 24000])]))
    assert sizes[:3] == [2, 2, 1] and np.allclose(emb, expected)  # input order kept after the length sort
