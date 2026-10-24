"""silero-vad: oip-v2 conformance with a mocked model, segment logic, and a real-model test when weights exist."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
vad_mod = load_server(SERVICE_DIR, "vad")
META = service_meta("silero-vad")


class FakeSession:
    """Stands in for the ONNX session: speech probability = frame RMS > 0.05 (with state passthrough)."""

    def run(self, _outputs, feeds):
        x, state = feeds["input"], feeds["state"]
        assert x.shape[0] == 1 and x.shape[1] in (512 + 64, 256 + 32), x.shape
        assert state.shape == (2, 1, 128) and feeds["sr"].dtype == np.int64
        frame = x[:, 64:] if x.shape[1] == 576 else x[:, 32:]
        prob = 0.95 if float(np.sqrt(np.mean(frame**2))) > 0.05 else 0.02
        return np.array([[prob]], dtype=np.float32), state


class FakeVAD(vad_mod.SileroVAD):
    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        self.session = FakeSession()


def speech_like(sr=16000, seconds=3.0, start=1.0, end=2.0):
    t = np.arange(int(sr * seconds)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    on = (t >= start) & (t < end)
    audio[on] = 0.5 * np.sin(2 * np.pi * 220 * t[on])
    return audio.astype(np.float32)


@pytest.fixture(scope="module")
def client():
    with running(app_mod.build_app(FakeVAD())) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_segments_over_http(client):
    audio = speech_like()
    body = {"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": audio.tolist()},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]},
    ], "parameters": {"speech_pad_ms": 0}}
    out = client.post("/v2/models/silero-vad/infer", json=body).json()["outputs"][0]
    assert out["name"] == "SEGMENTS" and out["shape"] == [1, 2]
    start, end = out["data"]
    assert abs(start - 1.0) < 0.04 and abs(end - 2.0) < 0.12


def test_bad_sample_rate_and_parameters_are_400(client):
    base = [{"name": "AUDIO", "datatype": "FP32", "shape": [512], "data": [0.0] * 512}]
    r = client.post("/v2/models/silero-vad/infer",
                    json={"inputs": base + [{"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [44100]}]})
    assert r.status_code == 400 and "SAMPLE_RATE" in r.json()["error"]
    r = client.post("/v2/models/silero-vad/infer", json={"parameters": {"bogus": 1},
                    "inputs": base + [{"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]})
    assert r.status_code == 400 and "bogus" in r.json()["error"]


def test_multiple_of_16k_is_decimated():
    vad = FakeVAD()
    vad.load()
    audio = speech_like(sr=48000)
    segs = vad.segments(audio, 48000, vad_mod.VadParams(speech_pad_ms=0))
    assert segs.shape == (1, 2) and abs(segs[0, 0] - 1.0) < 0.04


def test_segment_logic_matches_reference_rules():
    p = vad_mod.VadParams()  # 250 ms min speech, 100 ms min silence, 30 ms pad
    w = 512
    probs = np.array([0.0] * 10 + [0.9] * 20 + [0.0] * 2 + [0.9] * 20 + [0.0] * 20 + [0.9] * 3 + [0.0] * 10)
    segs = vad_mod.speech_segments(probs, 16000, len(probs) * w, p)
    # The 2-frame (64 ms) gap is shorter than min silence, so the first two bursts merge; the
    # 3-frame (96 ms) burst is shorter than min speech and is dropped.
    assert len(segs) == 1
    assert segs[0][0] == 10 * w - 480 and segs[0][1] == 52 * w + 480


WEIGHTS = Path(os.environ.get("SILERO_VAD_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/silero-vad/v6.2.3"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "silero_vad.onnx").is_file(), reason=f"no silero_vad.onnx in {WEIGHTS}")
def test_real_model_runs():
    pytest.importorskip("onnxruntime")
    vad = vad_mod.SileroVAD(WEIGHTS)
    vad.load()
    silence = vad.segments(np.zeros(16000, dtype=np.float32), 16000, vad_mod.VadParams())
    assert silence.shape == (0, 2)
    probs = vad.speech_probs(np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), 16000)
    assert probs.shape == (32,) and np.all((probs >= 0) & (probs <= 1))
