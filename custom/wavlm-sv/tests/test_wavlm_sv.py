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

    def embed(self, audio, sample_rate):
        x = emb_mod.prepare(audio, sample_rate)
        self.seen.append(len(x))
        v = np.zeros(DIM, np.float32)
        v[0], v[1] = 1.0, float(np.sqrt(np.mean(x ** 2)))
        return v / np.linalg.norm(v)


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


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


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
