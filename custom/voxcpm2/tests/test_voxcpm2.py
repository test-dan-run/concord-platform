"""voxcpm2: oip-v2 conformance with a fake model behind the real request path, parameters, and the real model when
weights and voxcpm exist."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
synth_mod = load_server(SERVICE_DIR, "synth")
META = service_meta("voxcpm2")


class FakeTorch:
    class cuda:  # noqa: N801 - mirrors torch.cuda
        @staticmethod
        def is_available():
            return False

    def __init__(self):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed


class FakeVoxCPM:
    """Records the generate() call; returns a 48 kHz tone whose length follows the text and pitch follows the seed."""

    def __init__(self, torch):
        self.torch = torch
        self.calls = []

    def generate(self, text, cfg_value, inference_timesteps, **kw):
        import soundfile as sf

        for key in ("prompt_wav_path", "reference_wav_path"):
            if key in kw:
                ref, sr = sf.read(kw[key], dtype="float32")
                kw[key] = (len(ref), sr)
        self.calls.append({"text": text, "cfg_value": cfg_value, "steps": inference_timesteps, **kw})
        t = np.arange(int(48000 * 0.1 * len(text.split()))) / 48000
        return (0.2 * np.sin(2 * np.pi * (100 + self.torch.seed % 100) * t)).astype(np.float32)


class FakeSynth(synth_mod.VoxCPM2Synth):
    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        self._torch = FakeTorch()
        self.model = FakeVoxCPM(self._torch)


@pytest.fixture(scope="module")
def synth():
    pytest.importorskip("soundfile")
    return FakeSynth()


@pytest.fixture(scope="module")
def client(synth):
    with running(app_mod.build_app(synth)) as c:
        yield c


def infer(client, text="Ok lah, we go first.", ref=None, ref_sr=16000, ref_text="this is my voice", seed=7, **params):
    ref = np.full(16000, 0.1, np.float32) if ref is None else ref
    body = {"inputs": [{"name": "TEXT", "datatype": "BYTES", "shape": [1], "data": [text]},
                       {"name": "REF_AUDIO", "datatype": "FP32", "shape": [len(ref)], "data": ref.tolist()},
                       {"name": "REF_SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [ref_sr]},
                       {"name": "REF_TEXT", "datatype": "BYTES", "shape": [1], "data": [ref_text]},
                       {"name": "SEED", "datatype": "INT64", "shape": [1], "data": [seed]}]}
    if params:
        body["parameters"] = params
    return client.post("/v2/models/voxcpm2/infer", json=body)


def outputs(r):
    assert r.status_code == 200, r.text
    return {o["name"]: o for o in r.json()["outputs"]}


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_ultimate_mode_passes_the_clip_as_prompt_and_reference(client, synth):
    out = outputs(infer(client, seed=7))
    assert out["SAMPLE_RATE"]["data"] == [48000] and len(out["AUDIO"]["data"]) == int(48000 * 0.1 * 5)
    call = synth.model.calls[-1]
    assert call["text"] == "Ok lah, we go first." and call["prompt_text"] == "this is my voice"
    assert call["prompt_wav_path"] == (16000, 16000) and call["reference_wav_path"] == (16000, 16000)
    assert call["cfg_value"] == 2.0 and call["steps"] == 10 and synth._torch.seed == 7


@pytest.mark.parametrize("mode, keys", [("continuation", {"prompt_wav_path", "prompt_text"}),
                                        ("reference", {"reference_wav_path"})])
def test_modes(client, synth, mode, keys):
    outputs(infer(client, mode=mode, cfg_value=1.5, inference_timesteps=20))
    call = synth.model.calls[-1]
    assert keys == set(call) - {"text", "cfg_value", "steps"}
    assert call["cfg_value"] == 1.5 and call["steps"] == 20


def test_reference_mode_needs_no_transcript(client):
    outputs(infer(client, ref_text="", mode="reference"))


def test_seed_changes_the_audio(client):
    a = outputs(infer(client, seed=1))["AUDIO"]["data"]
    b = outputs(infer(client, seed=2))["AUDIO"]["data"]
    assert a != b and a == outputs(infer(client, seed=1))["AUDIO"]["data"]


@pytest.mark.parametrize("kwargs", [{"text": "  "}, {"ref": np.zeros(100, np.float32)}, {"ref_sr": 4000},
                                    {"ref_text": ""}, {"mode": "sing"}, {"cfg_value": 0}, {"inference_timesteps": 0},
                                    {"style": "x"}, {"text": "x " * 1001}])
def test_bad_requests_are_400(client, kwargs):
    assert infer(client, **kwargs).status_code == 400


WEIGHTS = Path(os.environ.get("VOXCPM2_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/voxcpm2/32279ef"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
def test_real_model_clones_deterministically():
    pytest.importorskip("voxcpm")
    s = synth_mod.VoxCPM2Synth(WEIGHTS)
    s.load()
    t = np.arange(16000 * 3) / 16000
    ref = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    a, sr = s.synthesize("Hello there.", ref, 16000, "a test tone", seed=3)
    b, _ = s.synthesize("Hello there.", ref, 16000, "a test tone", seed=3)
    assert sr == 48000 and len(a) > sr // 4 and np.array_equal(a, b)
