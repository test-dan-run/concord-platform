"""silero-vad: parameters of the wrong type are a 400, not a 500 from the segment maths."""

from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
vad_mod = load_server(SERVICE_DIR, "vad")


class FakeVAD(vad_mod.SileroVAD):
    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        pass

    def segments(self, audio, sample_rate, p):
        return np.zeros((0, 2), np.float32)


BAD = [{"threshold": "0.5"}, {"threshold": None}, {"threshold": True}, {"neg_threshold": "0.2"},
       {"min_speech_duration_ms": "x"}, {"min_silence_duration_ms": None}, {"speech_pad_ms": [30]}]


@pytest.mark.parametrize("params", BAD, ids=str)
def test_wrong_types_are_value_errors(params):
    with pytest.raises(ValueError, match=f"{next(iter(params))} must be a number"):
        vad_mod.VadParams.from_parameters(params)


def test_numbers_and_a_null_neg_threshold_are_accepted():
    p = vad_mod.VadParams.from_parameters({"threshold": 0.6, "neg_threshold": None, "min_speech_duration_ms": 200.5,
                                           "speech_pad_ms": 0})
    assert (p.threshold, p.neg_threshold, p.min_speech_duration_ms, p.speech_pad_ms) == (0.6, None, 200.5, 0)


def test_wrong_type_over_http_is_400():
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [512], "data": [0.0] * 512},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
            "parameters": {"threshold": "0.5"}}
    with running(app_mod.build_app(FakeVAD())) as c:
        r = c.post("/v2/models/silero-vad/infer", json=body)
    assert r.status_code == 400 and "threshold" in r.json()["error"]
