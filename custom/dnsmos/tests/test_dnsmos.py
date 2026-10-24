"""dnsmos: oip-v2 conformance with a fake session, the reference windowing and calibration, real model if present."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
mos_mod = load_server(SERVICE_DIR, "dnsmos")
META = service_meta("dnsmos")


class FakeDnsmos(mos_mod.Dnsmos):
    """Raw scores grow with RMS, so louder clips score higher."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.windows = 0

    def load(self):
        self.session = object()

    def raw(self, segment):
        assert segment.shape == (int(mos_mod.INPUT_LENGTH * mos_mod.SR),)
        self.windows += 1
        r = float(np.sqrt(np.mean(segment ** 2)))
        return np.array([3.0 + r, 3.5 + r, 2.5 + r], dtype=np.float32)


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("soxr")
    with running(app_mod.build_app(FakeDnsmos())) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_windowing_follows_the_reference():
    sr = mos_mod.SR
    assert len(mos_mod.windows(np.ones(sr * 5))) == 1  # repeated to 10 s -> one window
    assert len(mos_mod.windows(np.ones(sr * 3))) == 3  # repeated to 12 s (as the reference does) -> three
    assert len(mos_mod.windows(np.ones(sr * 12))) == 3  # floor(12) - 9.01 + 1 = 3 hops of 1 s
    w = mos_mod.windows(np.arange(sr * 4, dtype=np.float32))[0]
    assert w[sr * 4] == 0 and w[sr * 4 + 1] == 1  # the clip repeats from its start


def test_calibration_polynomials():
    m = FakeDnsmos()
    m.load()
    ovrl, sig, bak = m.score(np.zeros(16000, np.float32), 16000)
    assert ovrl == pytest.approx(float(mos_mod.P_OVR(2.5)), abs=1e-5)
    assert sig == pytest.approx(float(mos_mod.P_SIG(3.0)), abs=1e-5)
    assert bak == pytest.approx(float(mos_mod.P_BAK(3.5)), abs=1e-5)


def test_batch_uses_lengths(client):
    a = np.zeros((2, 32000), np.float32)
    a[0, :16000] = 0.5
    a[1, :] = 0.5
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [2, 32000], "data": a.reshape(-1).tolist()},
                    {"name": "LENGTHS", "datatype": "INT32", "shape": [2], "data": [16000, 32000]},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]}
    outs = {o["name"]: o for o in client.post("/v2/models/dnsmos/infer", json=b).json()["outputs"]}
    assert outs["MOS"]["shape"] == [2]
    assert outs["MOS"]["data"][0] == pytest.approx(outs["MOS"]["data"][1], abs=1e-4)  # padding is ignored


def test_bad_lengths_are_400(client):
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [1, 4], "data": [0.0] * 4},
                    {"name": "LENGTHS", "datatype": "INT32", "shape": [1], "data": [9]},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]}
    assert client.post("/v2/models/dnsmos/infer", json=b).status_code == 400


WEIGHTS = Path(os.environ.get("DNSMOS_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/dnsmos/p835-82f1b17"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "sig_bak_ovr.onnx").is_file(), reason=f"no sig_bak_ovr.onnx in {WEIGHTS}")
def test_real_model_prefers_clean_over_noisy():
    pytest.importorskip("onnxruntime")
    m = mos_mod.Dnsmos(WEIGHTS)
    m.load()
    t = np.arange(16000 * 4) / 16000
    clean = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    noisy = clean + np.random.default_rng(0).normal(0, 0.2, len(t)).astype(np.float32)
    assert m.score(clean, 16000)[2] > m.score(noisy, 16000)[2]  # BAK: background noise
