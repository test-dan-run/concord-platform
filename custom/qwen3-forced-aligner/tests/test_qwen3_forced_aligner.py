"""qwen3-forced-aligner: oip-v2 conformance with a fake model behind the real slot-to-time logic, input checks,
and the real model when weights and qwen-asr exist."""

import os
import re
import types
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

    def load(self):
        self.wrapper = types.SimpleNamespace(timestamp_segment_time=80.0, aligner_processor=FakeProcessor())

    def slot_classes(self, audio, aligner_text):
        n = aligner_text.count("<timestamp>")
        idx = np.array([(i // 2) * 4 + (i % 2) * 3 for i in range(n)])
        if n >= 4:
            idx[2] = 0  # a non-monotonic slot the fix must repair
        prob = np.repeat(0.9 - 0.1 * np.arange(n // 2), 2)
        return idx, prob


@pytest.fixture(scope="module")
def client():
    with running(app_mod.build_app(FakeAligner())) as c:
        yield c


def infer(client, text="Ok lah, we go first.", language="en", seconds=2.0, sr=16000):
    audio = np.full(int(seconds * sr), 0.1, np.float32)
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": audio.tolist()},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]},
                       {"name": "TEXT", "datatype": "BYTES", "shape": [1], "data": [text]},
                       {"name": "LANGUAGE", "datatype": "BYTES", "shape": [1], "data": [language]}]}
    return client.post("/v2/models/qwen3-forced-aligner/infer", json=body)


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_words_times_and_scores(client):
    r = infer(client)
    assert r.status_code == 200, r.text
    out = {o["name"]: o for o in r.json()["outputs"]}
    assert out["WORDS"]["data"] == ["Ok", "lah", "we", "go", "first"]
    times = np.asarray(out["TIMES"]["data"], np.float32).reshape(out["TIMES"]["shape"])
    assert times.shape == (5, 2)
    assert times[0].tolist() == pytest.approx([0.0, 0.24]) and times[1].tolist() == pytest.approx([0.24, 0.56])
    assert np.all(np.diff(times.reshape(-1)) >= 0)
    assert out["SCORES"]["data"] == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5])


def test_text_without_words_gives_empty_outputs(client):
    out = {o["name"]: o for o in infer(client, text="... !").json()["outputs"]}
    assert out["WORDS"]["shape"] == [0] and out["TIMES"]["shape"] == [0, 2] and out["SCORES"]["shape"] == [0]


@pytest.mark.parametrize("lang, name", [("en", "English"), ("English", "English"), ("ZH", "Chinese")])
def test_language_names(lang, name):
    assert al_mod.language_name(lang) == name


@pytest.mark.parametrize("kwargs", [{"language": "ja"}, {"language": "ms"}, {"sr": 4000}, {"seconds": 301.0}])
def test_bad_requests_are_400(client, kwargs):
    assert infer(client, **kwargs).status_code == 400


WEIGHTS = Path(os.environ.get("ALIGNER_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/qwen3-forced-aligner/0.6b-c7cbfc2"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason=f"no model.safetensors in {WEIGHTS}")
def test_real_model_aligns_in_order():
    pytest.importorskip("qwen_asr")
    a = al_mod.QwenForcedAligner(WEIGHTS)
    a.load()
    t = np.arange(16000 * 2) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    words, times, scores = a.align(audio, 16000, "hello there", "en")
    assert words == ["hello", "there"] and times.shape == (2, 2) and np.all(np.diff(times.reshape(-1)) >= 0)
    assert np.all((scores >= 0) & (scores <= 1))
