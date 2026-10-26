"""qwen3-forced-aligner: oip-v2 conformance with a fake model behind the real slot-to-time logic, the batch signature
and the 0.1.0 single-clip request, input checks, and the real model when weights and qwen-asr exist."""

import json
import os
import re
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
al_mod = load_server(SERVICE_DIR, "aligner")
META = service_meta("qwen3-forced-aligner")


class FakeProcessor:
    def encode_timestamp(self, text, language):
        words = [w for w in (re.sub(r"[^\w' ]", " ", t) for t in text.split()) for w in w.split()]
        return words, "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"

    def fix_timestamp(self, data):
        return [float(v) for v in np.maximum.accumulate(np.asarray(data))]  # monotonic, like the package


class FakeAligner(al_mod.QwenForcedAligner):
    """Each word takes 4 slots of 80 ms (start, end); probabilities fall from 0.9 by word."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.batches = []

    def load(self):
        self.wrapper = types.SimpleNamespace(timestamp_segment_time=80.0, aligner_processor=FakeProcessor())

    def slot_classes_batch(self, rows):
        self.batches.append([len(r.audio) for r in rows])
        out = []
        for r in rows:
            n = r.prompt.count("<timestamp>")
            idx = np.array([(i // 2) * 4 + (i % 2) * 3 for i in range(n)])
            if n >= 4:
                idx[2] = 0  # a non-monotonic slot the fix must repair
            out.append((idx, np.repeat(0.9 - 0.1 * np.arange(n // 2), 2)))
        return out


@pytest.fixture(scope="module")
def aligner():
    return FakeAligner()


@pytest.fixture(scope="module")
def client(aligner):
    with running(app_mod.build_app(aligner)) as c:
        yield c


def infer(client, text="Ok lah, we go first.", language="en", seconds=2.0, sr=16000):
    """The 0.1.0 single-clip request: AUDIO [-1], no LENGTHS, TEXT and LANGUAGE [1]."""
    audio = np.full(int(seconds * sr), 0.1, np.float32)
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": audio.tolist()},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]},
                       {"name": "TEXT", "datatype": "BYTES", "shape": [1], "data": [text]},
                       {"name": "LANGUAGE", "datatype": "BYTES", "shape": [1], "data": [language]}]}
    return client.post("/v2/models/qwen3-forced-aligner/infer", json=body)


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def test_words_times_and_scores(client):
    r = infer(client)
    assert r.status_code == 200, r.text
    out = {o["name"]: o for o in r.json()["outputs"]}
    assert list(out) == ["WORDS", "TIMES", "SCORES"]  # the 0.1.0 outputs: no ROWS
    assert out["WORDS"]["data"] == ["Ok", "lah", "we", "go", "first"]
    times = np.asarray(out["TIMES"]["data"], np.float32).reshape(out["TIMES"]["shape"])
    assert times.shape == (5, 2)
    assert times[0].tolist() == pytest.approx([0.0, 0.24]) and times[1].tolist() == pytest.approx([0.24, 0.56])
    assert np.all(np.diff(times.reshape(-1)) >= 0)
    assert out["SCORES"]["data"] == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5])


def test_text_without_words_gives_empty_outputs(client):
    out = {o["name"]: o for o in infer(client, text="... !").json()["outputs"]}
    assert out["WORDS"]["shape"] == [0] and out["TIMES"]["shape"] == [0, 2] and out["SCORES"]["shape"] == [0]


@pytest.mark.parametrize("lang, name", [("English", "English"), ("ZH", "Chinese")])
def test_language_names(lang, name):
    assert al_mod.language_name(lang) == name


@pytest.mark.parametrize("kwargs", [{"language": "ja"}, {"sr": 4000}, {"seconds": 1.5}])
def test_bad_requests_are_400(client, monkeypatch, kwargs):
    if "seconds" in kwargs:
        monkeypatch.setattr(al_mod, "MAX_S", 1.0)  # an over-long clip of 1.5 s, not 180 s of JSON floats
    assert infer(client, **kwargs).status_code == 400


def infer_batch(client, clips, texts, languages, sr=16000, outputs=None, lengths=None):
    """The 0.2.0 batch request: AUDIO [N, longest] padded, LENGTHS [N], TEXT and LANGUAGE [N] or [1]."""
    audio = np.zeros((len(clips), max(len(c) for c in clips)), np.float32)
    for i, c in enumerate(clips):
        audio[i, : len(c)] = c
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [len(clips)],
                        "data": lengths or [len(c) for c in clips]},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]},
                       {"name": "TEXT", "datatype": "BYTES", "shape": [len(texts)], "data": texts},
                       {"name": "LANGUAGE", "datatype": "BYTES", "shape": [len(languages)], "data": languages}]}
    if outputs:
        body["outputs"] = [{"name": n} for n in outputs]
    return client.post("/v2/models/qwen3-forced-aligner/infer", json=body)


def outputs(r):
    assert r.status_code == 200, r.text
    return {o["name"]: np.asarray(o["data"]).reshape(o["shape"]) for o in r.json()["outputs"]}


def test_batch_gives_each_row_its_words(client):
    clips = [np.full(n, 0.1, np.float32) for n in (32000, 8000, 16000)]
    out = outputs(infer_batch(client, clips, ["one two", "...", "Ok lah, we go"], ["en", "en", "English"]))
    assert out["WORDS"].tolist() == ["one", "two", "Ok", "lah", "we", "go"]
    assert out["ROWS"].tolist() == [0, 0, 2, 2, 2, 2]  # row 1 has no words
    assert out["TIMES"].shape == (6, 2) and out["SCORES"].tolist() == pytest.approx([0.9, 0.8, 0.9, 0.8, 0.7, 0.6])
    assert out["TIMES"][2:4].reshape(-1).tolist() == pytest.approx([0.0, 0.24, 0.24, 0.56])  # each row's own times


def test_one_text_and_language_for_every_row(client):
    clips = [np.full(16000, 0.1, np.float32)] * 3
    out = outputs(infer_batch(client, clips, ["hello there"], ["en"]))
    assert out["ROWS"].tolist() == [0, 0, 1, 1, 2, 2] and out["WORDS"].tolist() == ["hello", "there"] * 3


def test_rows_run_in_batches_within_the_limit(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    a = FakeAligner()
    with running(app_mod.build_app(a)) as c:
        clips = [np.full(n, 0.1, np.float32) for n in (48000, 16000, 32000, 8000, 24000)]
        out = outputs(infer_batch(c, clips, ["a b", "c", "d e f", "g", "h"], ["en"]))
    assert a.batches == [[8000, 16000], [24000, 32000], [48000]]  # sorted by length
    assert out["WORDS"].tolist() == list("abcdefgh") and out["ROWS"].tolist() == [0, 0, 1, 2, 2, 2, 3, 4]


@pytest.mark.parametrize("texts, languages, lengths", [(["a", "b"], ["en"], None), (["a"], ["en", "en"], None),
                                                       (["a"], ["en"], [8000, 0, 8000]), (["a"], ["en"], [8000, 9999, 8000])])
def test_bad_batches_are_400(client, texts, languages, lengths):
    r = infer_batch(client, [np.full(8000, 0.1, np.float32)] * 3, texts, languages, lengths=lengths)
    assert r.status_code == 400, r.text


def test_single_clip_request_can_ask_for_rows(client):
    audio = np.full(16000, 0.1, np.float32)
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [16000], "data": audio.tolist()},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]},
                       {"name": "TEXT", "datatype": "BYTES", "shape": [1], "data": ["hello there"]},
                       {"name": "LANGUAGE", "datatype": "BYTES", "shape": [1], "data": ["en"]}],
            "outputs": [{"name": "WORDS"}, {"name": "ROWS"}]}
    out = outputs(client.post("/v2/models/qwen3-forced-aligner/versions/0.6b-c7cbfc2/infer", json=body))
    assert list(out) == ["WORDS", "ROWS"] and out["ROWS"].tolist() == [0, 0]


def test_aicore_client_sends_the_old_request_with_the_new_signature(client):
    """An app written for 0.1.0, with the 0.2.0 manifest signature in its endpoints map, still gets 0.1.0 outputs."""
    OIPClient = pytest.importorskip("aicore").OIPClient  # not installed in the image; PYTHONPATH=clients/aicore/src adds it

    c = OIPClient("http://testserver", META["name"], META["signature"], http=client)
    out = c.infer({"AUDIO": np.full(16000, 0.1, np.float32), "SAMPLE_RATE": np.array([16000], np.int32),
                   "TEXT": np.array(["hello there"]), "LANGUAGE": np.array(["en"])})
    assert list(out) == ["WORDS", "TIMES", "SCORES"] and out["WORDS"].tolist() == ["hello", "there"]


WEIGHTS = Path(os.environ.get("ALIGNER_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/qwen3-forced-aligner/0.6b-c7cbfc2"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
def test_real_model_aligns_in_order():
    al_mod._stub_nagisa()  # qwen_asr imports nagisa at module level; the image has no nagisa (see aligner.py)
    pytest.importorskip("qwen_asr")
    a = al_mod.QwenForcedAligner(WEIGHTS)
    a.load()
    t = np.arange(16000 * 2) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    words, times, scores = a.align(audio, 16000, "hello there", "en")
    assert words == ["hello", "there"] and times.shape == (2, 2) and np.all(np.diff(times.reshape(-1)) >= 0)
    assert np.all((scores >= 0) & (scores <= 1))


CHECK_CLIP = Path(__file__).resolve().parents[3] / "catalog" / "fixtures" / "check_clip.wav"


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
def test_real_model_batch_matches_one_by_one():
    """The check clip's three turns and the whole clip, aligned in one padded batch and one by one: the same words,
    times within one 80 ms step, and close scores."""
    al_mod._stub_nagisa()
    pytest.importorskip("qwen_asr")
    with wave.open(str(CHECK_CLIP)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    truth = json.loads(CHECK_CLIP.with_suffix(".json").read_text())
    clips = [(a[int(max(0, t["start"] - 0.2) * 16000): int((t["end"] + 0.2) * 16000)], t["text"]) for t in truth["turns"]]
    clips.append((a, truth["transcript"]))
    al = al_mod.QwenForcedAligner(WEIGHTS)
    al.load()
    rows = [al.row(x, 16000, text, "en") for x, text in clips]
    batched = al.align_rows(rows)
    for row, (words, times, scores) in zip(rows, batched, strict=True):
        w1, t1, s1 = al.align_rows([row])[0]
        assert words == w1 and len(words) > 5
        assert np.abs(times - t1).max() <= 0.081, np.abs(times - t1).max()
        assert np.abs(scores - s1).max() <= 0.05 and np.all(np.diff(times.reshape(-1)) >= 0)
        assert times.max() <= len(row.audio) / 16000 + 0.1
