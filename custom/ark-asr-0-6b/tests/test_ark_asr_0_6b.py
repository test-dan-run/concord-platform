"""ark-asr-0-6b: audio.transcriptions conformance with a fake model behind the real decoding and chunking, the authors'
text cleanup and token blocking, and the real model when weights exist."""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta
from conformance.checks import TONE_WAV

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
asr_mod = load_server(SERVICE_DIR, "ark")
META = service_meta("ark-asr-0-6b")


class FakeArk(asr_mod.ArkAsr):
    """One word per chunk, naming its length, run through the real cleanup (with a turn marker to cut)."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.chunks = []

    def load(self):
        self.model = object()

    def transcribe_chunk(self, x):
        self.chunks.append(len(x))
        return asr_mod.normalize_prediction_text(f", chunk{len(x)}<|im_end|><|user|>")


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("soundfile")
    return FakeArk()


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


def test_json_has_only_text(client):
    assert post(client, TONE_WAV.read_bytes()).json()["text"].startswith("chunk")
    r = post(client, wav(1.0))
    assert r.json() == {"text": "chunk16000"}


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


@pytest.mark.parametrize("raw, clean", [
    ("Hello  world.<|im_end|>", "Hello world."),
    ("<|start_content|>你好<|end_content|><|assistant|>junk", "你好"),
    ("prefix<|text|> - kept text", "kept text"),
    ("<|bicodec_semantic_12|>a <|bicodec_global_3|>b", "a b"),
    ("", ""),
])
def test_cleanup_matches_the_authors_script(raw, clean):
    assert asr_mod.normalize_prediction_text(raw) == clean


class FakeTokenizer:
    eos_token_id = 151645
    all_special_ids = [151643, 151645]
    added_tokens_decoder = {151663: {"content": "<|audio|>"}, 151664: {"content": "hello"},
                            151700: {"content": "<|bicodec_semantic_1|>"}}

    def convert_tokens_to_ids(self, token):
        return {"<|user|>": 151660, "<|assistant|>": 151661, "<|im_end|>": 151645}.get(token, -1)


def test_eos_and_blocked_ids():
    tok = FakeTokenizer()
    assert asr_mod.eos_token_ids(tok) == [151645, 151660, 151661]
    # specials and control tokens below 151670 are blocked, the eos id is kept, ids >= 151670 are left to the range
    assert asr_mod.blocked_token_ids(tok, keep={151645}) == [151643, 151663]


def test_block_processor_masks_range_and_ids():
    torch = pytest.importorskip("torch")
    scores = torch.zeros(1, 10)
    out = asr_mod.BlockTokenIds(8, [2, 99])(None, scores)
    assert torch.isinf(out[0, [2, 8, 9]]).all() and torch.isfinite(out[0, [0, 1, 3, 7]]).all()


WEIGHTS = Path(os.environ.get("ARK_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/ark-asr-0-6b/45776b5"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no weights in {WEIGHTS}")
def test_real_model_loads_and_transcribes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("librosa")
    m = asr_mod.ArkAsr(WEIGHTS)
    m.load()
    text, duration = m.transcribe(wav(1.0))
    assert isinstance(text, str) and duration == 1.0
