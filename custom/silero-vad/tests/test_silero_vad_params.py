"""silero-vad: parameters of the wrong type are a ValueError (a 400 over HTTP), not a 500 from the segment maths."""

from pathlib import Path

import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
vad_mod = load_server(SERVICE_DIR, "vad")


BAD = [{"threshold": "0.5"}, {"threshold": None}, {"threshold": True}, {"neg_threshold": "0.2"}]


@pytest.mark.parametrize("params", BAD, ids=str)
def test_wrong_types_are_value_errors(params):
    with pytest.raises(ValueError, match=f"{next(iter(params))} must be a number"):
        vad_mod.VadParams.from_parameters(params)


def test_numbers_a_null_neg_threshold_and_edge_values_are_accepted():
    p = vad_mod.VadParams.from_parameters({"threshold": 0.6, "neg_threshold": None, "min_speech_duration_ms": 200.5,
                                           "speech_pad_ms": 0})
    assert (p.threshold, p.neg_threshold, p.min_speech_duration_ms, p.speech_pad_ms) == (0.6, None, 200.5, 0)
    p = vad_mod.VadParams.from_parameters({"neg_threshold": 0.3, "min_speech_duration_ms": 0,
                                           "min_silence_duration_ms": 0, "speech_pad_ms": 0})
    assert (p.neg_threshold, p.min_speech_duration_ms, p.min_silence_duration_ms, p.speech_pad_ms) == (0.3, 0, 0, 0)
    assert vad_mod.VadParams.from_parameters({"speech_pad_ms": 5000}).speech_pad_ms == 5000
