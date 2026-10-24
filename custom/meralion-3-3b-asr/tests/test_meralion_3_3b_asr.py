"""meralion-3-3b-asr: audio.transcriptions conformance with a fake model behind the real decoding, chunking and
batching of concurrent requests, and the real model when weights exist."""

import io
import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta
from conformance.checks import TONE_WAV

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
asr_mod = load_server(SERVICE_DIR, "meralion")
META = service_meta("meralion-3-3b-asr")


class FakeMeralion(asr_mod.Meralion3Asr):
    """One word per chunk, naming its length; the first chunk gets the speaker tag the real model sometimes adds."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.chunks, self.batches = [], []

    def load(self):
        self.model = object()

    def transcribe_batch(self, pieces):
        self.batches.append([len(x) for x in pieces])
        out = []
        for x in pieces:
            self.chunks.append(len(x))
            out.append(asr_mod.strip_speaker_tag(("<Speaker1>: " if len(self.chunks) == 1 else "") + f"chunk{len(x)}"))
        return out


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("soundfile")
    return FakeMeralion()


@pytest.fixture(scope="module")
def client(model):
    with running(app_mod.build_app(model)) as c:
        yield c


def wav(seconds: float, sr: int = 16000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.full(int(seconds * sr), 0.1, np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post(client, payload, **data):
    files = {"file": ("a.wav", payload, "audio/wav")}
    return client.post("/v1/audio/transcriptions", data={"model": META["name"], **data}, files=files)


def post_together(client, payloads):
    """One thread per payload, all released at once; responses in payload order."""
    out, start = [None] * len(payloads), threading.Barrier(len(payloads))

    def call(i):
        start.wait()
        out[i] = post(client, payloads[i])

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(payloads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_json_has_only_text_and_the_tag_is_stripped(client):
    assert post(client, TONE_WAV.read_bytes()).json()["text"].startswith("chunk")
    r = post(client, wav(1.0))
    assert set(r.json()) == {"text"} and not r.json()["text"].startswith("<")


def test_long_audio_is_cut_into_30_s_chunks(client, model):
    model.chunks.clear()
    out = post(client, wav(65.0, sr=48000), response_format="verbose_json").json()
    assert model.chunks == [480000, 480000, 80000]  # resampled to 16 kHz, then 30 + 30 + 5 s
    assert out["text"] == "chunk480000 chunk480000 chunk80000" and out["duration"] == 65.0


def test_text_format(client):
    r = post(client, wav(1.0), response_format="text")
    assert r.status_code == 200 and r.text == "chunk16000"


def test_undecodable_audio_is_400(client):
    assert post(client, b"not audio at all").status_code == 400


def test_concurrent_requests_share_one_batch(monkeypatch):
    monkeypatch.setenv("AISVC_BATCH_WAIT_MS", "500")
    model = FakeMeralion()
    with running(app_mod.build_app(model)) as c:
        rs = post_together(c, [wav(1.0), wav(65.0), b"not audio", wav(3.0, sr=8000)])
    assert [r.status_code for r in rs] == [200, 200, 400, 200]  # a bad file fails only its own request
    assert [rs[i].json()["text"] for i in (0, 1, 3)] == ["chunk16000", "chunk480000 chunk480000 chunk80000", "chunk48000"]
    assert len(model.batches) == 1 and sorted(model.batches[0]) == [16000, 48000, 80000, 480000, 480000]


def test_chunks_run_in_batches_within_the_limit(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    model = FakeMeralion()
    with running(app_mod.build_app(model)) as c:
        out = post(c, wav(95.0)).json()
    assert model.batches == [[480000, 480000], [480000, 80000]]  # every chunk counts as 30 s, so the order is kept
    assert out["text"] == "chunk480000 chunk480000 chunk480000 chunk80000"


WEIGHTS = Path(os.environ.get("MERALION_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/meralion-3-3b-asr/9f226d7"))
CHECK_CLIP = Path(__file__).resolve().parents[3] / "catalog" / "fixtures" / "check_clip.wav"


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors.index.json").is_file(), reason=f"no weights in {WEIGHTS}")
def test_real_model_loads_and_transcribes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    m = asr_mod.Meralion3Asr(WEIGHTS)
    m.load()
    text, duration = m.transcribe(wav(1.0))
    assert isinstance(text, str) and duration == 1.0


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors.index.json").is_file(), reason=f"no weights in {WEIGHTS}")
def test_real_model_batch_matches_one_by_one():
    """Greedy decoding of a batch of chunks of different lengths gives each chunk its own text."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from catalog.verifier.metrics import wer

    with wave.open(str(CHECK_CLIP)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    pieces = [a, a[: 16000 * 4], a[16000 * 3 : 16000 * 9], np.tile(a, 3)[: 16000 * 28]]
    m = asr_mod.Meralion3Asr(WEIGHTS)
    m.load()
    batched = m.transcribe_batch(pieces)
    alone = [m.transcribe_batch([p])[0] for p in pieces]
    assert all(t for t in alone), alone
    for one, b in zip(alone, batched, strict=True):
        assert one == b or wer(one, b) <= 0.05, (one, b)
