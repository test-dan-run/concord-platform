"""deepfilternet3: parameters of the wrong type are a ValueError (a 400 over HTTP, see test_bad_parameters_are_400),
and post_filter must be a real true or false."""

from pathlib import Path

import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
df_mod = load_server(SERVICE_DIR, "deepfilter")


@pytest.mark.parametrize("params, message", [
    ({"atten_lim_db": "20"}, "atten_lim_db must be a number"),
    ({"atten_lim_db": True}, "atten_lim_db must be a number"),
    ({"post_filter": "no"}, "post_filter must be true or false"),
], ids=str)
def test_wrong_types_are_value_errors(params, message):
    with pytest.raises(ValueError, match=message):
        df_mod.DfParams.from_parameters(params)
