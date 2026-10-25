"""ark-asr-0-6b: audio.transcriptions conformance with a fake model behind the real decoding, chunking and batching
of concurrent requests, the authors' text cleanup and token blocking, the encoder wrapper, and the real model when
weights exist."""

import io
import json
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
asr_mod = load_server(SERVICE_DIR, "ark")
META = service_meta("ark-asr-0-6b")


class FakeArk(asr_mod.ArkAsr):
    """One word per chunk, naming its length, run through the real cleanup (with a turn marker to cut)."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.chunks, self.batches = [], []

    def load(self):
        self.model = object()

    def transcribe_batch(self, pieces):
        self.batches.append([len(x) for x in pieces])
        self.chunks += [len(x) for x in pieces]
        return [asr_mod.normalize_prediction_text(f", chunk{len(x)}<|im_end|><|user|>") for x in pieces]


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
    assert sorted(model.chunks) == [80000, 480000, 480000]  # resampled to 16 kHz, then 30 + 30 + 5 s (run shortest first)
    assert out["text"] == "chunk480000 chunk480000 chunk80000" and out["duration"] == 65.0  # joined in file order


def test_a_tail_under_0_1_s_is_dropped():
    x = np.zeros(480016, dtype=np.float32)  # 30 s and 16 samples: the STFT cannot pad a 16-sample chunk
    assert [len(c) for c in asr_mod.chunks(x)] == [480000]
    assert [len(c) for c in asr_mod.chunks(np.zeros(481600, dtype=np.float32))] == [480000, 1600]
    assert [len(c) for c in asr_mod.chunks(np.zeros(1600, dtype=np.float32))] == [1600]  # a 0.1 s file is kept
    assert asr_mod.chunks(np.zeros(1599, dtype=np.float32)) == []  # a shorter file has no pieces


def test_a_file_under_0_1_s_gives_empty_text_without_the_model(client, model):
    model.chunks.clear()
    r = post(client, wav(0.001), response_format="verbose_json")  # 16 samples
    assert r.status_code == 200 and r.json()["text"] == "" and r.json()["duration"] == 0.001
    assert model.chunks == []


def test_text_format(client):
    r = post(client, wav(1.0), response_format="text")
    assert r.status_code == 200 and r.text == "chunk16000"


def test_undecodable_audio_is_400(client):
    assert post(client, b"not audio at all").status_code == 400


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


def test_concurrent_requests_share_one_batch(monkeypatch):
    monkeypatch.setenv("AISVC_BATCH_WAIT_MS", "500")
    model = FakeArk()
    with running(app_mod.build_app(model)) as c:
        rs = post_together(c, [wav(1.0), wav(65.0), b"not audio", wav(3.0, sr=8000)])
    assert [r.status_code for r in rs] == [200, 200, 400, 200]  # a bad file fails only its own request
    assert [rs[i].json()["text"] for i in (0, 1, 3)] == ["chunk16000", "chunk480000 chunk480000 chunk80000", "chunk48000"]
    assert model.batches == [[16000, 48000, 80000, 480000, 480000]]  # one generate, sorted by length


def test_chunks_run_in_batches_within_the_limits(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "3")
    monkeypatch.setenv("AISVC_MAX_BATCH_LEN", "60000")  # ms: 3 x 20 s, or 2 x 30 s
    model = FakeArk()
    with running(app_mod.build_app(model)) as c:
        out = post(c, wav(95.0)).json()
    assert model.batches == [[80000, 480000], [480000, 480000]]
    assert out["text"] == "chunk480000 chunk480000 chunk480000 chunk80000"


def test_encoder_wrapper_encodes_each_row_at_its_own_length():
    """A padded row must give what it gives alone; the stand-in encoder (mean over all frames) shows any padding."""
    torch = pytest.importorskip("torch")

    class MeanEncoder(torch.nn.Module):  # (B, mel, T) -> (B, T // 4, 2): a function of every frame, like attention
        def forward(self, mel):
            m = mel.mean(dim=(1, 2), keepdim=True).squeeze(1)
            return m.unsqueeze(-1).expand(mel.shape[0], mel.shape[-1] // 4, 2) + 0

    enc = asr_mod.own_lengths(torch, MeanEncoder())
    rows = [torch.rand(1, 3, n) for n in (8, 16, 8)]
    mel = torch.zeros(3, 3, 16)
    for i, r in enumerate(rows):
        mel[i, :, : r.shape[-1]] = r[0]
    enc.frames = [8, 16, 8]
    out = enc(mel)
    assert out.shape == (3, 4, 2)
    for i, r in enumerate(rows):
        alone = MeanEncoder()(r)[0]
        assert torch.allclose(out[i, : alone.shape[0]], alone) and not out[i, alone.shape[0]:].any()
    enc.frames = None
    assert torch.allclose(enc(mel), MeanEncoder()(mel))  # unset: the plain encoder


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


CHECK_CLIP = Path(__file__).resolve().parents[3] / "catalog" / "fixtures" / "check_clip.wav"


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no weights in {WEIGHTS}")
def test_real_model_batch_matches_one_by_one():
    """Greedy decoding of a batch of chunks of different lengths (left-padded prompts, each chunk encoded at its own
    length) gives each chunk its own text."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("librosa")
    from catalog.verifier.metrics import wer

    with wave.open(str(CHECK_CLIP)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    pieces = [a, a[: 16000 * 4], a[16000 * 3 : 16000 * 9], np.tile(a, 3)[: 16000 * 28], a[: 16000 * 4]]
    m = asr_mod.ArkAsr(WEIGHTS)
    m.load()
    batched = m.transcribe_batch(pieces)
    alone = [m.transcribe_batch([p])[0] for p in pieces]
    assert all(t for t in alone), alone
    truth = json.loads(CHECK_CLIP.with_suffix(".json").read_text())["transcript"]
    assert wer(truth, alone[0]) < 0.5, alone[0]  # the verifier's warning level: the model really hears the clip
    for one, b in zip(alone, batched, strict=True):
        assert one == b or wer(one, b) <= 0.05, (one, b)
