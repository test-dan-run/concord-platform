"""deepfilternet3: oip-v2 conformance with a fake enhancer, parameters, and a real-binary test when weights exist."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
df_mod = load_server(SERVICE_DIR, "deepfilter")
META = service_meta("deepfilternet3")


class FakeDeepFilter(df_mod.DeepFilter):
    """Halves the signal; records the parameters it was called with."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        pass

    def enhance(self, audio, sample_rate, p):
        self.calls.append((sample_rate, p))
        return (np.asarray(audio, np.float32) * 0.5).astype(np.float32)


@pytest.fixture(scope="module")
def fake():
    return FakeDeepFilter()


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


def test_audio_round_trip_keeps_rate_and_length(client, fake):
    audio = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    r = client.post("/v2/models/deepfilternet3/infer", json=body(audio, 8000, {"atten_lim_db": 20}))
    outs = {o["name"]: o for o in r.json()["outputs"]}
    assert outs["AUDIO"]["shape"] == [800] and outs["SAMPLE_RATE"]["data"] == [8000]
    assert fake.calls[-1] == (8000, df_mod.DfParams(atten_lim_db=20))


def test_bad_parameters_are_400(client):
    r = client.post("/v2/models/deepfilternet3/infer", json=body(np.zeros(16), parameters={"bogus": 1}))
    assert r.status_code == 400 and "bogus" in r.json()["error"]
    r = client.post("/v2/models/deepfilternet3/infer", json=body(np.zeros(16), parameters={"atten_lim_db": 300}))
    assert r.status_code == 400


WEIGHTS = Path(os.environ.get("DEEPFILTERNET3_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/deepfilternet3/v0.5.6"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "deep-filter").is_file(), reason=f"no deep-filter binary in {WEIGHTS}")
def test_real_binary_denoises():
    pytest.importorskip("soxr")
    pytest.importorskip("soundfile")
    m = df_mod.DeepFilter(WEIGHTS)
    m.load()
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 2) / 16000
    speechy = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    noisy = speechy + rng.normal(0, 0.05, len(t)).astype(np.float32)
    out = m.enhance(noisy, 16000, df_mod.DfParams())
    assert out.shape == noisy.shape and out.dtype == np.float32
    quiet = np.sin(2 * np.pi * 3 * t) <= 0  # noise-only half
    assert np.std(out[quiet]) < 0.5 * np.std(noisy[quiet])
