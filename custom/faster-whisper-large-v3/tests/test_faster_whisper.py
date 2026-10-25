"""faster-whisper-large-v3: audio.transcriptions conformance (mocked model), verbose_json/word timestamps, the batch
size and its out of memory backoff, and the real model when present."""

import importlib.util
import os
import wave
from pathlib import Path

import numpy as np
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
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=StubModel,
                                                                             BatchedInferencePipeline=None))
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


@pytest.fixture
def pipeline(monkeypatch):
    """Stub BatchedInferencePipeline: records each batch_size, runs out of memory above `fits` windows."""
    import sys
    import types

    calls, fits = [], [64]

    class StubPipeline:
        def __init__(self, model):
            self.model = model

        def transcribe(self, audio, batch_size, **kw):
            calls.append((batch_size, kw))

            def segments():
                if batch_size > fits[0]:
                    raise RuntimeError("CUDA failed with error out of memory")
                yield types.SimpleNamespace(text=" hi")
            return segments(), types.SimpleNamespace(language="en", duration=len(audio) / 16000)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(BatchedInferencePipeline=StubPipeline))
    monkeypatch.setitem(sys.modules, "torch", None)  # a CTranslate2 image: no torch
    return calls, fits


@pytest.mark.parametrize("env, size", [({}, 8), ({"AISVC_MAX_BATCH": "16"}, 16),
                                       ({"AISVC_MAX_BATCH": "16", "AISVC_MAX_BATCH_LEN": "100000"}, 3),
                                       ({"AISVC_MAX_BATCH_LEN": "1000"}, 1)])
def test_batch_size_from_the_limits(monkeypatch, env, size):
    for k in ("AISVC_MAX_BATCH", "AISVC_MAX_BATCH_LEN"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert model_mod.FasterWhisper("/nonexistent").batch_size() == size  # windows of 30 s


def test_out_of_memory_halves_the_batch_and_keeps_it(pipeline, monkeypatch):
    calls, fits = pipeline
    monkeypatch.setenv("AISVC_MAX_BATCH", "16")
    m = model_mod.FasterWhisper("/nonexistent")
    m.model = object()
    fits[0] = 5
    segments, info = m.run_batched(np.zeros(16000, np.float32), language="en", without_timestamps=False)
    assert [c[0] for c in calls] == [16, 8, 4] and [s.text for s in segments] == [" hi"] and info.language == "en"
    assert calls[-1][1] == {"language": "en", "without_timestamps": False}
    assert m.batch_size() == 4 and m.limits.stats()["backoffs"] == 2  # the lower limit stays for later requests
    m.run_batched(np.zeros(16000, np.float32))
    assert calls[-1][0] == 4


def test_out_of_memory_on_one_window_raises(pipeline, monkeypatch):
    calls, fits = pipeline
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    m = model_mod.FasterWhisper("/nonexistent")
    m.model = object()
    fits[0] = 0
    with pytest.raises(RuntimeError, match="out of memory"):
        m.run_batched(np.zeros(16000, np.float32))
    assert [c[0] for c in calls] == [2, 1]


def test_other_errors_are_not_retried(pipeline, monkeypatch):
    calls, _ = pipeline
    m = model_mod.FasterWhisper("/nonexistent")
    m.model = object()
    with pytest.raises(TypeError):
        m.run_batched(None)  # len(None) fails inside the stub
    assert len(calls) == 1


def test_limits_are_reported_on_platform_memory(client):
    assert {"max_items", "max_len", "backoffs"} <= set(client.get("/platform/memory").json()["batch"]["transcribe"])


WEIGHTS = Path(os.environ.get("WHISPER_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "faster-whisper-large-v3"))
CHECK_CLIP = Path(__file__).resolve().parents[3] / "catalog" / "fixtures" / "check_clip.wav"


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


def _long_clip(copies: int = 6) -> np.ndarray:
    """The 10.56 s check clip six times with 1 s of silence between: about 70 s, several VAD windows."""
    with wave.open(str(CHECK_CLIP)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return np.concatenate([np.concatenate([a, np.zeros(16000, np.float32)]) for _ in range(copies)])


@pytest.mark.weights
@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is None, reason="faster-whisper not installed")
@pytest.mark.skipif(not (WEIGHTS / "model.bin").is_file(), reason=f"no model.bin in {WEIGHTS}")
def test_real_model_batched_matches_sequential():
    """Batched windows against a batch of one (padding only: every window is 30 s) and against the sequential
    decoder (which also conditions on the previous text). Word timestamps must still come back in order."""
    from catalog.verifier.metrics import wer

    m = model_mod.FasterWhisper(WEIGHTS)
    m.load()
    audio = _long_clip()
    kw = {"language": "en", "word_timestamps": True, "vad_filter": True, "beam_size": 5, "without_timestamps": False}
    text = lambda segs: "".join(s.text for s in segs)  # noqa: E731
    batched, info = m.run_batched(audio, **kw)
    m.limits.max_items = 1
    one, _ = m.run_batched(audio, **kw)
    m.limits.max_items = m.limits.configured[0]
    seq = list(m.model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True, beam_size=5)[0])
    assert wer(text(one), text(batched)) <= 0.02, (text(one), text(batched))
    assert wer(text(seq), text(batched)) <= 0.1, (text(seq), text(batched))
    words = [w for s in batched for w in s.words]
    seq_words = sum(len(s.words) for s in seq)
    assert abs(len(words) - seq_words) <= 0.1 * seq_words and len(batched) >= 6  # finer than one segment per window
    assert all(0 <= w.start <= w.end <= info.duration + 0.5 for w in words)
    assert all(a.start <= b.start + 0.01 for a, b in zip(words, words[1:]))
