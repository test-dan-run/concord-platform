"""kokoro-tts: audio.speech conformance with a mocked model (no torch, weights or GPU needed)."""

import io
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
model_mod = load_server(SERVICE_DIR, "model")
META = service_meta("kokoro-tts")


class FakeKokoro(model_mod.KokoroModel):
    """Real voice resolution against a fake voices dir; synthesis is a short sine."""

    def __init__(self, weights_dir: Path, gate: threading.Event | None = None):
        super().__init__(weights_dir)
        self.gate = gate
        self.calls = []

    def load(self):
        if self.gate:
            self.gate.wait(5)
        self.pipelines = {lang: object() for lang in model_mod.LANGS}

    def synthesize(self, text, voice, speed=1.0):
        self.resolve_voice(voice)
        self.calls.append((text, voice, speed))
        t = np.arange(int(model_mod.SAMPLE_RATE * 0.25 / speed)) / model_mod.SAMPLE_RATE
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), model_mod.SAMPLE_RATE


@pytest.fixture(scope="module")
def weights_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("kokoro")
    (d / "voices").mkdir()
    for v in ("af_alloy", "af_heart", "am_echo", "bm_george", "jf_alpha"):
        (d / "voices" / f"{v}.pt").write_bytes(b"")
    return d


@pytest.fixture(scope="module")
def model(weights_dir):
    return FakeKokoro(weights_dir)


@pytest.fixture(scope="module")
def client(model):
    with running(app_mod.build_app(model)) as c:
        yield c


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def test_wav_is_24khz_mono(client):
    r = client.post("/v1/audio/speech", json={"model": "kokoro-tts", "input": "hi", "voice": "af_heart", "response_format": "wav"})
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)
        assert w.getnframes() == 6000


def test_pcm_is_raw_int16(client):
    r = client.post("/v1/audio/speech", json={"model": "kokoro-tts", "input": "hi", "voice": "alloy", "response_format": "pcm"})
    assert r.headers["content-type"] == "audio/pcm" and len(r.content) == 6000 * 2


def test_openai_voice_names_and_speed_reach_the_model(client, model):
    client.post("/v1/audio/speech", json={"model": "kokoro-tts", "input": "x", "voice": "echo", "speed": 2.0, "response_format": "wav"})
    assert model.calls[-1] == ("x", "echo", 2.0)
    assert model.resolve_voice("echo")[1].name == "am_echo.pt"
    assert model.resolve_voice("bm_george")[0] == "b"


@pytest.mark.parametrize("fmt,ctype", [("mp3", "audio/mpeg")])  # flac takes the same soundfile path
def test_compressed_formats(client, fmt, ctype):
    pytest.importorskip("soundfile")
    r = client.post("/v1/audio/speech", json={"model": "kokoro-tts", "input": "hi", "voice": "alloy", "response_format": fmt})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == ctype and len(r.content) > 100


@pytest.mark.parametrize("body,status,code", [
    ({"voice": "jf_alpha"}, 400, None),             # Japanese voice: no Japanese pipeline in the image
    ({"response_format": "aac"}, 400, None),        # no aac encoder
    ({"model": "tts-1"}, 404, "model_not_found"),
    ({"speed": 9}, 400, "invalid_request"),
    ({"input": ""}, 400, "invalid_request"),
])
def test_errors_are_openai_shaped(client, body, status, code):
    req = {"model": "kokoro-tts", "input": "hello", "voice": "alloy", "response_format": "wav"} | body
    r = client.post("/v1/audio/speech", json=req)
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert set(err) >= {"message", "type", "code"} and err["message"]
    if code:
        assert err["code"] == code


def test_health_is_503_while_loading(weights_dir):
    from fastapi.testclient import TestClient

    gate = threading.Event()
    app = app_mod.build_app(FakeKokoro(weights_dir, gate))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/v1/audio/speech", json={"model": "kokoro-tts", "input": "x", "voice": "alloy"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        gate.set()
        assert app.state.svc.wait(5)
        assert c.get("/health").status_code == 200
