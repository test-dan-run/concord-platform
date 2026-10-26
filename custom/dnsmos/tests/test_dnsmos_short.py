"""dnsmos: a clip of one sample at 44.1 or 48 kHz, which resampling to 16 kHz turns into no samples."""

from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
mos_mod = load_server(SERVICE_DIR, "dnsmos")


class FakeDnsmos(mos_mod.Dnsmos):
    def load(self):
        self.session = object()

    def raw_batch(self, segments):
        return np.full((len(segments), 3), 3.0, np.float32)


def test_one_sample_keeps_that_sample():
    sr = 44100  # 48 kHz takes the same path; the request test below uses it
    pytest.importorskip("soxr")
    x = mos_mod.prepare(np.ones(1, np.float32), sr)
    assert x.tolist() == [1.0]
    assert len(mos_mod.windows(x)) == 1  # repeated to one 9.01 s window


def test_one_sample_request_is_200():
    sr = 48000
    pytest.importorskip("soxr")
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [1, 1], "data": [0.1]},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [1], "data": [1]},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    with running(app_mod.build_app(FakeDnsmos("/nonexistent"))) as c:
        r = c.post("/v2/models/dnsmos/infer", json=body)
    assert r.status_code == 200, r.text
