"""pyannote-diarization: oip-v2 conformance with a mocked model, shared diarization tensors, real model when present."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
diar_mod = load_server(SERVICE_DIR, "diarizer")
turns_mod = load_server(SERVICE_DIR, "turns")
META = service_meta("pyannote-diarization")


class FakeDiarizer(diar_mod.PyannoteDiarizer):
    """Two speakers, returned out of time order with labels that are not in arrival order."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        self.model = self.pipeline = object()

    def diarize(self, audio, sample_rate, *args):
        self.calls.append((len(audio), sample_rate, args))
        if sample_rate != 16000:
            raise ValueError("SAMPLE_RATE must be 16000")
        if not np.any(audio):
            return turns_mod.turns_to_tensors([])
        return turns_mod.turns_to_tensors([(5.0, 7.5, "B"), (0.5, 4.0, "A"), (8.0, 9.0, "B"), (3.5, 5.5, "A")])


@pytest.fixture(scope="module")
def fake():
    return FakeDiarizer()


@pytest.fixture(scope="module")
def client(fake):
    with running(app_mod.build_app(fake)) as c:
        yield c


def infer(client, audio, sr=16000, **extra):
    return client.post("/v2/models/pyannote-diarization/infer", json={"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}], **extra})


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_turns_sorted_and_speakers_in_arrival_order(client):
    out = {o["name"]: o for o in infer(client, np.ones(1600)).json()["outputs"]}
    assert out["SEGMENTS"]["shape"] == [4, 2] and out["SPEAKERS"]["shape"] == [4]
    assert out["SEGMENTS"]["data"][::2] == [0.5, 3.5, 5.0, 8.0]
    assert out["SPEAKERS"]["data"] == [0, 0, 1, 1]  # "A" spoke first -> 0


def test_silence_gives_empty_tensors(client):
    out = {o["name"]: o for o in infer(client, np.zeros(1600)).json()["outputs"]}
    assert out["SEGMENTS"]["shape"] == [0, 2] and out["SPEAKERS"]["shape"] == [0]


def test_model_errors_are_400(client):
    r = infer(client, np.ones(800), sr=8000)
    assert r.status_code == 400 and "SAMPLE_RATE" in r.json()["error"]


def test_turns_to_tensors_unit():
    t = turns_mod.turns_to_tensors([(2.0, 3.0, "spk9"), (0.0, 1.0, "spk3"), (1.5, 2.5, "spk9")])
    assert t["SEGMENTS"].dtype == np.float32 and t["SPEAKERS"].dtype == np.int32
    assert t["SEGMENTS"].tolist() == [[0.0, 1.0], [1.5, 2.5], [2.0, 3.0]] and t["SPEAKERS"].tolist() == [0, 1, 1]


WEIGHTS = Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets"))) / "pyannote-speaker-diarization-3.1"


@pytest.fixture(scope="module")
def real():
    if importlib.util.find_spec("pyannote") is None:
        pytest.skip("pyannote.audio not installed (runs inside the image or a matching venv)")
    if not (WEIGHTS / "config.yaml").is_file():
        pytest.skip(f"no config.yaml in {WEIGHTS}")
    d = diar_mod.PyannoteDiarizer(WEIGHTS)
    d.load()
    return d


def test_parameters_are_validated_and_passed(client, fake):
    r = infer(client, np.ones(1600), parameters={"num_speakers": 2, "exclusive": True})
    assert r.status_code == 200 and fake.calls[-1][2][0] == diar_mod.DiarParams(num_speakers=2, exclusive=True)
    for bad in ({"num_speakers": 0}, {"speakers": 2}, {"exclusive": "yes"}):
        r = infer(client, np.ones(1600), parameters=bad)
        assert r.status_code == 400, bad


@pytest.mark.weights
def test_real_model(real):
    out = real.diarize(np.zeros(16000 * 3, np.float32), 16000, diar_mod.DiarParams())
    assert out["SEGMENTS"].shape == (0, 2)
    wav = os.environ.get("DIARIZATION_TEST_WAV")  # e.g. a two-speaker 16 kHz clip
    if wav:
        import soundfile as sf

        audio, sr = sf.read(wav, dtype="float32")
        for p in (diar_mod.DiarParams(), diar_mod.DiarParams(exclusive=True, max_speakers=4)):
            out = real.diarize(audio, sr, p)
            assert len(set(out["SPEAKERS"].tolist())) >= 2 and out["SPEAKERS"][0] == 0
        out8k = real.diarize(audio[::2], sr // 2, diar_mod.DiarParams())  # resampled by pyannote
        assert len(set(out8k["SPEAKERS"].tolist())) >= 2
