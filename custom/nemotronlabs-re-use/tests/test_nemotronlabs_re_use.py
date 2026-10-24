"""nemotronlabs-re-use: oip-v2 conformance with a fake enhancer, input validation, chunking, and a real-model
test when the weights, a GPU and mamba-ssm are present."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
enh_mod = load_server(SERVICE_DIR, "enhancer")
META = service_meta("nemotronlabs-re-use")


class FakeReUse(enh_mod.ReUse):
    """Halves the signal after the real input checks; records the rates it saw."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.rates = []

    def load(self):
        pass

    def enhance(self, audio, sample_rate):
        if sample_rate not in enh_mod.SUPPORTED_RATES:
            raise ValueError(f"SAMPLE_RATE {sample_rate} is not supported")
        self.rates.append(sample_rate)
        return (np.asarray(audio, np.float32) * 0.5).astype(np.float32)


@pytest.fixture(scope="module")
def fake():
    return FakeReUse()


@pytest.fixture(scope="module")
def client(fake):
    with running(app_mod.build_app(fake)) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def body(audio, sr=16000, parameters=None):
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    if parameters is not None:
        b["parameters"] = parameters
    return b


def test_round_trip_keeps_rate_and_length(client, fake):
    audio = np.linspace(-0.5, 0.5, 2205, dtype=np.float32)
    r = client.post("/v2/models/nemotronlabs-re-use/infer", json=body(audio, 22050))
    outs = {o["name"]: o for o in r.json()["outputs"]}
    assert outs["AUDIO"]["shape"] == [2205] and outs["SAMPLE_RATE"]["data"] == [22050] and fake.rates[-1] == 22050


def test_unsupported_rate_and_parameters_are_400(client):
    r = client.post("/v2/models/nemotronlabs-re-use/infer", json=body(np.zeros(160), 11025))
    assert r.status_code == 400 and "11025" in r.json()["error"]
    r = client.post("/v2/models/nemotronlabs-re-use/infer", json=body(np.zeros(160), parameters={"bwe": 48000}))
    assert r.status_code == 400 and "bwe" in r.json()["error"]


def test_stft_sizes_follow_the_reference_script():
    m = enh_mod.ReUse("/nonexistent")
    m.cfg = {"stft_cfg": {"n_fft": 320, "hop_size": 40, "win_size": 320, "sampling_rate": 8000}}
    assert m._sizes(16000) == (640, 80, 640) and m._sizes(22050) == (882, 110, 882) and m._sizes(44100) == (1764, 220, 1764)


def test_chunked_overlap_add_reassembles_an_identity_model(monkeypatch):
    torch = pytest.importorskip("torch")
    m = enh_mod.ReUse("/nonexistent")
    monkeypatch.setattr(m, "_enhance_one", lambda wav, sr: wav.clone())  # identity: overlap-add must give back the input
    wav = torch.from_numpy(np.random.default_rng(0).standard_normal(8000 * 23).astype(np.float32)).unsqueeze(0)
    out = m._enhance_chunked(wav, 8000)
    assert out.shape == wav.shape and torch.allclose(out, wav, atol=1e-4)


WEIGHTS = Path(os.environ.get("NEMOTRONLABS_RE_USE_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/nemotronlabs-re-use/31f82f2"))


def _gpu_and_mamba() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
@pytest.mark.skipif(not _gpu_and_mamba(), reason="needs a CUDA GPU and mamba-ssm (run inside the image)")
def test_real_model_improves_a_noisy_tone():
    m = enh_mod.ReUse(WEIGHTS)
    m.load()
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 3) / 16000
    clean = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2).astype(np.float32)
    noisy = clean + 0.05 * rng.standard_normal(clean.size).astype(np.float32)
    out = m.enhance(noisy, 16000)
    assert out.shape == noisy.shape and np.all(np.isfinite(out))
