"""faster-whisper-large-v3: audio.transcriptions conformance (mocked model), verbose_json/word timestamps, real model when present."""

import importlib.util
import os
from pathlib import Path

import pytest
from conformance import checks_for, load_server, running, service_meta
from conformance.checks import TONE_WAV

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
model_mod = load_server(SERVICE_DIR, "model")
META = service_meta("faster-whisper-large-v3")


class FakeWhisper(model_mod.FasterWhisper):
    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        self.model = object()

    def transcribe(self, audio_bytes, **kw):
        if not audio_bytes.startswith(b"RIFF"):
            raise ValueError("could not decode audio file: invalid data")
        self.calls.append(kw)
        words = [{"word": " hello", "start": 0.1, "end": 0.4, "probability": 0.9},
                 {"word": " world", "start": 0.5, "end": 0.9, "probability": 0.8}]
        return {"task": "transcribe", "language": "en", "duration": 1.0, "text": "hello world",
                "segments": [{"id": 1, "seek": 0, "start": 0.1, "end": 0.9, "text": " hello world", "tokens": [1, 2],
                              "temperature": 0.0, "avg_logprob": -0.2, "compression_ratio": 1.0, "no_speech_prob": 0.01}],
                "words": words if kw.get("word_timestamps") else []}


@pytest.fixture(scope="module")
def model():
    return FakeWhisper()


@pytest.fixture(scope="module")
def client(model):
    with running(app_mod.build_app(model)) as c:
        yield c


def post(client, data, payload=None):
    files = {"file": ("a.wav", payload if payload is not None else TONE_WAV.read_bytes(), "audio/wav")}
    return client.post("/v1/audio/transcriptions", data={"model": "faster-whisper-large-v3", **data}, files=files)


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_json_has_only_text(client):
    assert post(client, {}).json() == {"text": "hello world"}


def test_text_format(client):
    r = post(client, {"response_format": "text"})
    assert r.text == "hello world" and r.headers["content-type"].startswith("text/plain")


def test_verbose_json_segments_without_words(client, model):
    out = post(client, {"response_format": "verbose_json"}).json()
    assert out["segments"][0]["start"] == 0.1 and "words" not in out and out["language"] == "en"
    assert model.calls[-1]["word_timestamps"] is False


def test_word_timestamps(client, model):
    out = post(client, {"response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}).json()
    assert [w["word"] for w in out["words"]] == [" hello", " world"]
    assert model.calls[-1]["word_timestamps"] is True


def test_options_reach_the_model(client, model):
    post(client, {"language": "de", "prompt": "Glossar: Diarisierung", "temperature": "0.2"})
    assert model.calls[-1] | {"word_timestamps": None} == {"language": "de", "prompt": "Glossar: Diarisierung",
                                                         "temperature": 0.2, "word_timestamps": None}


def test_undecodable_audio_is_400(client):
    r = post(client, {}, payload=b"not audio")
    assert r.status_code == 400 and "could not decode" in r.json()["error"]["message"]


def test_bad_granularity_is_400(client):
    assert post(client, {"timestamp_granularities[]": "char"}).status_code == 400


@pytest.fixture
def fake_ct2(monkeypatch, tmp_path):
    """Stub ctranslate2 + faster_whisper; returns (weights dir, opened devices, set of devices that fail)."""
    import sys
    import types

    (tmp_path / "model.bin").write_bytes(b"")
    opened, broken, gpus = [], set(), [1]

    class StubModel:
        def __init__(self, path, device, compute_type):
            opened.append((device, compute_type))
            self.device = device

        def transcribe(self, audio, **kw):
            if self.device in broken:
                raise RuntimeError("Library libcublas.so.12 is not found")
            return iter([]), None

    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: gpus[0]))
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=StubModel))
    monkeypatch.delenv("ASR_DEVICE", raising=False)
    monkeypatch.delenv("ASR_COMPUTE_TYPE", raising=False)
    return tmp_path, opened, broken, gpus


def test_defaults_to_gpu(fake_ct2):
    weights, opened, _, _ = fake_ct2
    m = model_mod.FasterWhisper(weights)
    m.load()
    assert m.device == "cuda" and opened == [("cuda", "float16")]


def test_falls_back_to_cpu_when_cuda_fails(fake_ct2):
    weights, opened, broken, _ = fake_ct2
    broken.add("cuda")
    m = model_mod.FasterWhisper(weights)
    m.load()
    assert m.device == "cpu" and opened == [("cuda", "float16"), ("cpu", "int8")]


def test_cpu_when_no_gpu(fake_ct2):
    weights, opened, _, gpus = fake_ct2
    gpus[0] = 0
    m = model_mod.FasterWhisper(weights)
    m.load()
    assert m.device == "cpu" and opened == [("cpu", "int8")]


def test_asr_device_forces_device(fake_ct2, monkeypatch):
    weights, opened, _, _ = fake_ct2
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    m = model_mod.FasterWhisper(weights)
    m.load()
    assert m.device == "cpu" and opened == [("cpu", "int8")]


WEIGHTS = Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets"))) / "faster-whisper-large-v3"


@pytest.mark.weights
@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is None, reason="faster-whisper not installed")
@pytest.mark.skipif(not (WEIGHTS / "model.bin").is_file(), reason=f"no model.bin in {WEIGHTS}")
def test_real_model():
    m = model_mod.FasterWhisper(WEIGHTS)
    m.load()
    wav = Path(os.environ.get("ASR_TEST_WAV", TONE_WAV))
    out = m.transcribe(wav.read_bytes(), word_timestamps=True, language="en")
    assert isinstance(out["text"], str) and out["duration"] > 0.5
    assert all(w["start"] <= w["end"] for w in out["words"])
    if "ASR_TEST_WAV" in os.environ:
        assert len(out["words"]) > 5, out["text"]
