"""qwen3-forced-aligner: a row under 0.1 s gets no words without running the model, so it cannot fail the batch
(the Whisper feature extractor's STFT fails on 200 samples or fewer, as it did in ark-asr)."""

import types
from pathlib import Path

import numpy as np
import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
al_mod = load_server(SERVICE_DIR, "aligner")


class Processor:
    def encode_timestamp(self, text, language):
        words = text.split()
        return words, "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"

    def fix_timestamp(self, data):
        return [float(v) for v in data]


class Aligner(al_mod.QwenForcedAligner):
    """Records the rows the model gets; two slots per word at class 0 and 1."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.wrapper = types.SimpleNamespace(timestamp_segment_time=80.0, aligner_processor=Processor())
        self.ran = []

    def slot_classes_batch(self, rows):
        self.ran.append([len(r.audio) for r in rows])
        return [(np.tile([0, 1], len(r.words)), np.ones(2 * len(r.words), np.float32)) for r in rows]


def test_a_row_under_0_1_s_gives_empty_without_the_model():
    a = Aligner()
    row = a.row(np.full(1599, 0.1, np.float32), 16000, "hello there", "en")  # 100 samples: see the test below
    words, times, scores = a.align_rows([row])[0]
    assert words == [] and times.shape == (0, 2) and scores.shape == (0,)
    assert a.ran == []


def test_short_rows_leave_the_others_in_the_batch():
    a = Aligner()
    rows = [a.row(np.full(n, 0.1, np.float32), 16000, "hello there", "en") for n in (16000, 100, 1600)]
    out = a.align_rows(rows)
    assert [r[0] for r in out] == [["hello", "there"], [], ["hello", "there"]]
    assert a.ran == [[16000, 1600]]  # 0.1 s itself still runs


def test_a_short_row_at_48_khz_counts_its_16_khz_samples():
    pytest.importorskip("soxr")
    a = Aligner()
    row = a.row(np.full(4797, 0.1, np.float32), 48000, "hello", "en")  # just under 0.1 s
    assert a.align_rows([row])[0][0] == [] and a.ran == []
