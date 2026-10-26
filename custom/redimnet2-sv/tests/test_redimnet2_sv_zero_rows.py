"""redimnet2-sv: a request with no rows is a clear 400 before the model runs."""

from pathlib import Path

import numpy as np
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
emb_mod = load_server(SERVICE_DIR, "embedder")


class FakeEmbedder(emb_mod.ReDimNet2Embedder):
    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = 0

    def load(self):
        self.model = object()

    def embed_batch(self, clips):
        self.calls += 1
        return [np.ones(4, np.float32) / 2 for _ in clips]


def test_zero_rows_is_400_with_a_clear_message():
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [0, 0], "data": []},
                       {"name": "LENGTHS", "datatype": "INT32", "shape": [0], "data": []},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]}
    model = FakeEmbedder()
    with running(app_mod.build_app(model)) as c:
        r = c.post("/v2/models/redimnet2-sv/infer", json=body)
    assert r.status_code == 400 and r.json() == {"error": "AUDIO has no rows"}, r.text
    assert model.calls == 0
