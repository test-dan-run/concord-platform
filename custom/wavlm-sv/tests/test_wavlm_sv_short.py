"""wavlm-sv: a clip of one sample at 44.1 or 48 kHz, which resampling to 16 kHz turns into no samples."""

from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
emb_mod = load_server(SERVICE_DIR, "embedder")


class FakeEmbedder(emb_mod.WavLMEmbedder):
    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        self.model = object()

    def embed_batch(self, clips):
        return [np.ones(4, np.float32) / 2 for _ in clips]


@pytest.mark.parametrize("sr", [44100, 48000])
def test_one_sample_is_repeated_to_the_minimum_length(sr):
    pytest.importorskip("soxr")
    x = emb_mod.prepare(np.ones(1, np.float32), sr)
    assert len(x) == int(emb_mod.MIN_S * emb_mod.SR) and np.all(x == 1.0)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_one_sample_request_is_200(sr):
    pytest.importorskip("soxr")
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [1, 1], "data": [0.1]},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [1], "data": [1]},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]}
    with running(app_mod.build_app(FakeEmbedder())) as c:
        r = c.post("/v2/models/wavlm-sv/infer", json=body)
    assert r.status_code == 200, r.text
