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


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def body(audio, sr=16000, parameters=None):
    """The single-clip request that callers sent before the batch signature."""
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    if parameters is not None:
        b["parameters"] = parameters
    return b


def test_single_clip_request_keeps_rate_length_and_the_old_shapes(client, fake):
    audio = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    r = client.post("/v2/models/deepfilternet3/infer", json=body(audio, 8000, {"atten_lim_db": 20}))
    outs = {o["name"]: o for o in r.json()["outputs"]}
    assert sorted(outs) == ["AUDIO", "SAMPLE_RATE"] and outs["AUDIO"]["shape"] == [800] and outs["SAMPLE_RATE"]["data"] == [8000]
    assert fake.calls[-1] == (8000, df_mod.DfParams(atten_lim_db=20))


def test_batch_rows_run_one_by_one_and_come_back_padded(client, fake):
    clips = [np.full(n, 0.2 * (k + 1), np.float32) for k, n in enumerate([300, 1600, 0, 800])]
    audio = np.zeros((4, 1600), np.float32)
    for row, c in zip(audio, clips):
        row[: len(c)] = c
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [4, 1600], "data": audio.reshape(-1).tolist()},
                    {"name": "LENGTHS", "datatype": "INT32", "shape": [4], "data": [300, 1600, 0, 800]},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
         "parameters": {"post_filter": True}}
    del fake.calls[:]
    outs = {o["name"]: o for o in client.post("/v2/models/deepfilternet3/infer", json=b).json()["outputs"]}
    assert outs["AUDIO"]["shape"] == [4, 1600] and outs["LENGTHS"]["data"] == [300, 1600, 0, 800]
    assert fake.calls == [(16000, df_mod.DfParams(post_filter=True))] * 4
    got = np.asarray(outs["AUDIO"]["data"], np.float32).reshape(4, 1600)
    for row, c in zip(got, clips):
        assert np.allclose(row[: len(c)], c * 0.5) and not np.any(row[len(c):])


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
    for n in (8, 300):  # the CLI crashes on very short input unless it is padded first
        assert m.enhance(noisy[:n], 16000, df_mod.DfParams()).shape == (n,)
