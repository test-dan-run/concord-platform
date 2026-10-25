"""wpe-dereverb: parameters that are not integers are a 400, and a fraction is not cut to an integer."""

from pathlib import Path

import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
wpe_mod = load_server(SERVICE_DIR, "dereverb")


@pytest.mark.parametrize("params", [{"taps": None}, {"taps": "x"}, {"taps": "10"}, {"taps": 2.7}, {"delay": True},
                                    {"iterations": [3]}], ids=str)
def test_wrong_types_are_value_errors(params):
    with pytest.raises(ValueError, match=f"{next(iter(params))} must be an integer"):
        wpe_mod.WpeParams.from_parameters(params)


def test_integers_are_accepted():
    assert wpe_mod.WpeParams.from_parameters({"taps": 5, "delay": 2, "iterations": 1}) == wpe_mod.WpeParams(5, 2, 1)


def test_wrong_type_over_http_is_400():
    pytest.importorskip("nara_wpe")
    body = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [4], "data": [0.0] * 4},
                       {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
            "parameters": {"taps": None}}
    with running(app_mod.build_app()) as c:
        r = c.post("/v2/models/wpe-dereverb/infer", json=body)
    assert r.status_code == 400 and "taps" in r.json()["error"]
