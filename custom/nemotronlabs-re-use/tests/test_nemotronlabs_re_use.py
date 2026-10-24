"""nemotronlabs-re-use: oip-v2 conformance with a fake enhancer, input validation, chunking and batching, and
real-model tests when the weights, a GPU and mamba-ssm are present."""

import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
enh_mod = load_server(SERVICE_DIR, "enhancer")
META = service_meta("nemotronlabs-re-use")
URL = "/v2/models/nemotronlabs-re-use/infer"


class FakeReUse(enh_mod.ReUse):
    """Halves each piece; records the rates and the pieces it saw."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.rates, self.calls = [], []

    def load(self):
        pass

    def enhance_batch(self, pieces):
        self.calls.append([len(x) for x, _ in pieces])
        self.rates.extend(sr for _, sr in pieces)
        return [(np.asarray(x, np.float32) * 0.5).astype(np.float32) for x, _ in pieces]


@pytest.fixture(scope="module")
def fake():
    return FakeReUse()


@pytest.fixture(scope="module")
def client(fake):
    with running(app_mod.build_app(fake)) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def body(audio, sr=16000, parameters=None):
    """The single-clip request that callers sent before the batch signature."""
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    if parameters is not None:
        b["parameters"] = parameters
    return b


def batch_body(clips, sr=16000):
    from aisvc.batch import pad

    audio, lengths = pad(clips)
    return {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [len(lengths)], "data": lengths.tolist()},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}


def test_single_clip_request_keeps_rate_length_and_the_old_shapes(client, fake):
    audio = np.linspace(-0.5, 0.5, 2205, dtype=np.float32)
    r = client.post(URL, json=body(audio, 22050))
    outs = {o["name"]: o for o in r.json()["outputs"]}
    assert sorted(outs) == ["AUDIO", "SAMPLE_RATE"] and outs["AUDIO"]["shape"] == [2205]
    assert outs["SAMPLE_RATE"]["data"] == [22050] and fake.rates[-1] == 22050
    assert np.allclose(outs["AUDIO"]["data"], audio * 0.5, atol=1e-6)
    r = client.post("/v2/models/nemotronlabs-re-use/versions/1/infer", json={**body(audio[:10]), "id": "x",
                                                                             "outputs": [{"name": "AUDIO"}]})
    assert r.json()["id"] == "x" and [(o["name"], o["shape"]) for o in r.json()["outputs"]] == [("AUDIO", [10])]
    r = client.post("/v2/models/nemotronlabs-re-use/versions/2/infer", json=body(audio[:10]))
    assert r.status_code == 404  # the route still checks the version


def test_batch_rows_come_back_padded_with_their_lengths(client):
    clips = [np.full(n, 0.2 * (k + 1), np.float32) for k, n in enumerate([300, 1600, 800])]
    r = client.post(URL, json=batch_body(clips))
    outs = {o["name"]: o for o in r.json()["outputs"]}
    assert outs["AUDIO"]["shape"] == [3, 1600] and outs["LENGTHS"]["data"] == [300, 1600, 800]
    audio = np.asarray(outs["AUDIO"]["data"], np.float32).reshape(3, 1600)
    for row, clip in zip(audio, clips):
        assert np.allclose(row[: len(clip)], clip * 0.5) and not np.any(row[len(clip):])


def test_unsupported_rate_bad_lengths_and_parameters_are_400(client):
    r = client.post(URL, json=body(np.zeros(160), 11025))
    assert r.status_code == 400 and "11025" in r.json()["error"]
    r = client.post(URL, json=body(np.zeros(160), parameters={"bwe": 48000}))
    assert r.status_code == 400 and "bwe" in r.json()["error"]
    b = batch_body([np.zeros(10, np.float32)])
    b["inputs"][1]["data"] = [11]
    assert client.post(URL, json=b).status_code == 400
    r = client.post(URL, json=batch_body([np.zeros(10, np.float32), np.zeros(0, np.float32)]))
    assert r.status_code == 400 and "empty" in r.json()["error"]


def test_stft_sizes_follow_the_reference_script():
    m = enh_mod.ReUse("/nonexistent")
    m.cfg = {"stft_cfg": {"n_fft": 320, "hop_size": 40, "win_size": 320, "sampling_rate": 8000}}
    assert m._sizes(16000) == (640, 80, 640) and m._sizes(22050) == (882, 110, 882) and m._sizes(44100) == (1764, 220, 1764)


def test_long_input_is_cut_into_equal_chunks_and_put_back_together():
    m = enh_mod.ReUse("/nonexistent")
    assert m.spans(240_000, 16000) == [(0, 240_000)]
    spans = m.spans(8000 * 31, 8000)  # 248k samples: over the one-pass limit, so 5 s chunks at a 2.5 s hop
    assert spans[0] == (0, 40000) and spans[-2] == (200000, 240000) and spans[-1] == (208000, 248000)
    assert {e - s for s, e in spans} == {40000} and len(spans) == 12
    x = np.random.default_rng(0).standard_normal(8000 * 31).astype(np.float32)
    out = m.enhance_rows([x], 8000, run=lambda pieces: [p.copy() for p, _ in pieces])  # identity model
    assert out[0].shape == x.shape and np.allclose(out[0], x, atol=1e-4)


def test_pieces_run_in_batches_within_the_limit_in_input_order(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    model = FakeReUse()
    rng = np.random.default_rng(3)
    clips = [rng.standard_normal(n).astype(np.float32) for n in (400, 16000 * 16, 900, 400)]  # the 16 s row is chunked
    with running(app_mod.build_app(model)) as c:
        r = c.post(URL, json=batch_body(clips))
    assert r.status_code == 200, r.text
    outs = {o["name"]: o for o in r.json()["outputs"]}
    audio = np.asarray(outs["AUDIO"]["data"], np.float32).reshape(4, -1)
    for row, n, clip in zip(audio, outs["LENGTHS"]["data"], clips, strict=True):
        assert n == len(clip) and np.allclose(row[:n], clip * 0.5, atol=1e-5)
    pieces = len(enh_mod.ReUse.spans(16000 * 16, 16000))
    assert all(len(call) <= 2 for call in model.calls) and sum(map(len, model.calls)) == 3 + pieces
    assert model.calls[0] == [400, 400]  # shortest first: the two equal rows share a batch


WEIGHTS = Path(os.environ.get("NEMOTRONLABS_RE_USE_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/nemotronlabs-re-use/31f82f2"))


def _gpu_and_mamba() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.fixture(scope="module")
def real():
    m = enh_mod.ReUse(WEIGHTS)
    m.load()
    return m


def _tone(seconds, sr, f0, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    clean = (0.3 * np.sin(2 * np.pi * f0 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2).astype(np.float32)
    return clean, clean + 0.05 * rng.standard_normal(clean.size).astype(np.float32)


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
@pytest.mark.skipif(not _gpu_and_mamba(), reason="needs a CUDA GPU and mamba-ssm (run inside the image)")
def test_real_model_improves_a_noisy_tone(real):
    clean, noisy = _tone(3, 16000, 220, 1)
    out = real.enhance(noisy, 16000)
    assert out.shape == noisy.shape and np.all(np.isfinite(out))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
@pytest.mark.skipif(not _gpu_and_mamba(), reason="needs a CUDA GPU and mamba-ssm (run inside the image)")
@pytest.mark.parametrize("sr", [16000, 48000])
def test_real_model_batch_matches_one_piece_at_a_time(real, sr):
    """Pieces of equal length share a forward pass; the output must match running each piece alone."""
    rows = [_tone(s, sr, f, k)[1] for k, (s, f) in enumerate([(1.3, 140), (4.0, 210), (4.0, 120), (0.01, 180), (21.0, 160)])]
    alone = real.enhance_rows(rows, sr, run=lambda pieces: [real.enhance_batch([p])[0] for p in pieces])
    batched = real.enhance_rows(rows, sr)
    for a, b in zip(alone, batched, strict=True):
        assert a.shape == b.shape and np.linalg.norm(a - b) <= 1e-3 * np.linalg.norm(a) + 1e-6, \
            float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
