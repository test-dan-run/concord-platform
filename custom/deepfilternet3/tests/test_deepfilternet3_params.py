"""deepfilternet3: parameters of the wrong type are a 400, and post_filter must be a real true or false."""

from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
df_mod = load_server(SERVICE_DIR, "deepfilter")


class FakeDeepFilter(df_mod.DeepFilter):
    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        pass

    def enhance(self, audio, sample_rate, p):
        self.calls.append(p)
        return np.asarray(audio, np.float32)


@pytest.mark.parametrize("params, message", [
    ({"atten_lim_db": None}, "atten_lim_db must be a number"),
    ({"atten_lim_db": "x"}, "atten_lim_db must be a number"),
    ({"atten_lim_db": "20"}, "atten_lim_db must be a number"),
    ({"atten_lim_db": True}, "atten_lim_db must be a number"),
    ({"post_filter": "no"}, "post_filter must be true or false"),
    ({"post_filter": 1}, "post_filter must be true or false"),
    ({"post_filter": None}, "post_filter must be true or false"),
], ids=str)
def test_wrong_types_are_value_errors(params, message):
    with pytest.raises(ValueError, match=message):
        df_mod.DfParams.from_parameters(params)


def test_numbers_and_booleans_are_accepted():
    assert df_mod.DfParams.from_parameters({"atten_lim_db": 20, "post_filter": True}) == df_mod.DfParams(20.0, True)
    assert df_mod.DfParams.from_parameters({"atten_lim_db": 12.5, "post_filter": False}).atten_lim_db == 12.5


@pytest.mark.parametrize("params", [{"atten_lim_db": None}, {"post_filter": "no"}], ids=str)
def test_wrong_type_over_http_is_400_and_never_runs(params):
    fake = FakeDeepFilter()
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [16], "data": [0.0] * 16},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
            "parameters": params}
    with running(app_mod.build_app(fake)) as c:
        r = c.post("/v2/models/deepfilternet3/infer", json=body)
    assert r.status_code == 400 and next(iter(params)) in r.json()["error"]
    assert fake.calls == []
