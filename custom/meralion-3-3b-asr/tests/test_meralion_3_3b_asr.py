"""meralion-3-3b-asr: audio.transcriptions conformance with a fake model behind the real decoding and chunking,
and the real model when weights exist."""

import io
import os
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
        self.chunks = []

    def load(self):
        self.model = object()

    def transcribe_chunk(self, x):
        self.chunks.append(len(x))
        return asr_mod.strip_speaker_tag(("<Speaker1>: " if len(self.chunks) == 1 else "") + f"chunk{len(x)}")


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


WEIGHTS = Path(os.environ.get("MERALION_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/meralion-3-3b-asr/9f226d7"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors.index.json").is_file(), reason=f"no weights in {WEIGHTS}")
def test_real_model_loads_and_transcribes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    m = asr_mod.Meralion3Asr(WEIGHTS)
    m.load()
    text, duration = m.transcribe(wav(1.0))
    assert isinstance(text, str) and duration == 1.0
