"""silero-vad: parameter values out of range are a 400, not segments past the clip or a 500 from the segment maths."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
vad_mod = load_server(SERVICE_DIR, "vad")

DURATIONS = ["min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"]


class FakeVAD(vad_mod.SileroVAD):
    """Real segment maths on fixed probabilities: speech from 0.5 s to 1.5 s of a 2 s clip."""

    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        pass

    def speech_probs(self, audio, sample_rate):
        frames = -(-len(audio) // 512)
        t = np.arange(frames) * 512 / sample_rate
        return ((t >= 0.5) & (t < 1.5)).astype(np.float32)


@pytest.mark.parametrize("name", DURATIONS)
@pytest.mark.parametrize("value", [-1, -2000, -1e308, math.inf, -math.inf, math.nan])
def test_durations_out_of_range_are_value_errors(name, value):
    with pytest.raises(ValueError, match=name):
        vad_mod.VadParams.from_parameters({name: value})


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, -0.1, 1.5, math.inf, math.nan])
def test_neg_threshold_out_of_range_is_a_value_error(value):
    with pytest.raises(ValueError, match="neg_threshold"):
        vad_mod.VadParams.from_parameters({"neg_threshold": value})


def test_edge_values_are_accepted():
    p = vad_mod.VadParams.from_parameters({"neg_threshold": 0.3, "min_speech_duration_ms": 0,
                                           "min_silence_duration_ms": 0, "speech_pad_ms": 0})
    assert (p.neg_threshold, p.min_speech_duration_ms, p.min_silence_duration_ms, p.speech_pad_ms) == (0.3, 0, 0, 0)
    assert vad_mod.VadParams.from_parameters({"speech_pad_ms": 5000}).speech_pad_ms == 5000


def post(client, raw_params: str):
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [32000], "data": [0.0] * 32000},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]}
    raw = json.dumps(body).replace('"inputs"', f'"parameters": {raw_params}, "inputs"', 1)
    return client.post("/v2/models/silero-vad/infer", content=raw, headers={"content-type": "application/json"})


@pytest.mark.parametrize("raw_params", ['{"speech_pad_ms": -2000}', '{"speech_pad_ms": -1e308}',
                                        '{"speech_pad_ms": Infinity}', '{"min_silence_duration_ms": -5}',
                                        '{"neg_threshold": 1.2}'])
def test_out_of_range_over_http_is_400(raw_params):
    with running(app_mod.build_app(FakeVAD())) as c:
        r = post(c, raw_params)
    assert r.status_code == 400, r.text
    assert next(iter(json.loads(raw_params.replace("Infinity", "1")))) in r.json()["error"]


def test_segments_stay_inside_the_clip():
    with running(app_mod.build_app(FakeVAD())) as c:
        r = post(c, '{"speech_pad_ms": 5000}')
    assert r.status_code == 200, r.text
    assert r.json()["outputs"][0]["data"] == [0.0, 2.0]
