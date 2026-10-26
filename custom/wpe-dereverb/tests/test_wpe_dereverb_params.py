"""wpe-dereverb: parameters that are not integers are a ValueError (a 400 over HTTP), and a fraction is not cut to an
integer."""

from pathlib import Path

import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
wpe_mod = load_server(SERVICE_DIR, "dereverb")


@pytest.mark.parametrize("params", [{"taps": "10"}, {"taps": 2.7}, {"delay": True}], ids=str)
def test_wrong_types_are_value_errors(params):
    with pytest.raises(ValueError, match=f"{next(iter(params))} must be an integer"):
        wpe_mod.WpeParams.from_parameters(params)


def test_integers_are_accepted():
    assert wpe_mod.WpeParams.from_parameters({"taps": 5, "delay": 2, "iterations": 1}) == wpe_mod.WpeParams(5, 2, 1)
