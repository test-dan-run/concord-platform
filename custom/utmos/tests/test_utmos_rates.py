"""utmos: a SAMPLE_RATE that resampling to 16 kHz cannot handle in reasonable memory is a 400, before the model runs."""

from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
mos_mod = load_server(SERVICE_DIR, "utmos")


class FakeUtmos(mos_mod.Utmos):
    def __init__(self):
        super().__init__("/nonexistent")
        self.seen = []

    def load(self):
        self.model = object()

    def score_batch(self, clips):
        self.seen.extend(sr for _, sr in clips)
        return [3.0] * len(clips)


def infer(client, sr):
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [1, 4], "data": [0.0, 0.1, -0.1, 0.0]},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [1], "data": [4]},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    return client.post("/v2/models/utmos/infer", json=body)


@pytest.mark.parametrize("sr", [44101, 192001, 16001])
def test_odd_rate_is_400_and_the_model_is_not_called(sr):
    model = FakeUtmos()
    with running(app_mod.build_app(model)) as c:
        r = infer(c, sr)
    assert r.status_code == 400, r.text
    assert "SAMPLE_RATE" in r.json()["error"]
    assert model.seen == []


COMMON = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000]


def test_common_rates_are_accepted():
    model = FakeUtmos()
    with running(app_mod.build_app(model)) as c:
        codes = {sr: infer(c, sr).status_code for sr in COMMON}
    assert codes == dict.fromkeys(COMMON, 200)
    assert model.seen == COMMON


def test_low_rate_message_is_unchanged():
    with running(app_mod.build_app(FakeUtmos())) as c:
        r = infer(c, 4000)
    assert r.status_code == 400 and "SAMPLE_RATE must be at least 8000, got 4000" in r.json()["error"]
