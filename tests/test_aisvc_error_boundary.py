"""aisvc: an error while the factory shapes a handler's result is a 500 in the contract's error shape, not a
plain-text 500, and a ValueError there is the server's error, not a 400."""

import numpy as np
import pytest
from aisvc import create_oip_app, create_openai_app
from fastapi.testclient import TestClient

CHAT = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
FORM = {"data": {"model": "m"}, "files": {"file": ("a.wav", b"1234", "audio/wav")}}


def post(app, path, **kwargs):
    with TestClient(app, raise_server_exceptions=False) as c:
        return c.post(path, headers={"origin": "http://localhost:5173"}, **kwargs)


@pytest.mark.parametrize("api, result, path, kwargs, message", [
    ("chat.completions", {"text": "hi"}, "/v1/chat/completions", {"json": CHAT}, "KeyError: 'content'"),
    ("chat.completions", 7, "/v1/chat/completions", {"json": CHAT}, "TypeError"),
    ("completions", 7, "/v1/completions", {"json": {"model": "m", "prompt": "x"}}, "TypeError"),
    ("embeddings", [["abc"]], "/v1/embeddings", {"json": {"model": "m", "input": "x"}}, "ValueError"),
    ("audio.transcriptions", {"language": "en"}, "/v1/audio/transcriptions", FORM, "KeyError: 'text'"),
], ids=lambda v: v if isinstance(v, str) and v.startswith("/") else None)
def test_openai_result_errors_are_openai_shaped_500s(api, result, path, kwargs, message):
    r = post(create_openai_app("m", api, lambda req: result, background_load=False), path, **kwargs)
    assert r.status_code == 500 and r.headers["content-type"] == "application/json", r.text
    err = r.json()["error"]
    assert err["type"] == "server_error" and err["message"].startswith(message)
    assert r.headers["access-control-allow-origin"] == "*"  # answered inside the app, so CORS still applies


SIG = {"inputs": [{"name": "X", "datatype": "FP32", "shape": [-1]}],
       "outputs": [{"name": "Y", "datatype": "FP32", "shape": [-1]}]}
X = {"inputs": [{"name": "X", "datatype": "FP32", "shape": [1], "data": [0.0]}]}


@pytest.mark.parametrize("result, message", [
    ({"Y": np.array(["a", "b"])}, "ValueError: could not convert string to float"),
    ({"Y": [[1.0], [2.0, 3.0]]}, "ValueError"),
    (None, "TypeError"),
], ids=["strings", "ragged", "none"])
def test_oip_result_errors_are_oip_shaped_500s(result, message):
    r = post(create_oip_app("m", SIG, lambda x, p: result, background_load=False), "/v2/models/m/infer", json=X)
    assert r.status_code == 500 and r.headers["content-type"] == "application/json", r.text
    assert r.json()["error"].startswith(message)


def test_oip_missing_output_and_wrong_shape_keep_their_messages():
    r = post(create_oip_app("m", SIG, lambda x, p: {}, background_load=False), "/v2/models/m/infer", json=X)
    assert r.status_code == 500 and r.json() == {"error": "handler did not produce output 'Y'"}
    r = post(create_oip_app("m", SIG, lambda x, p: {"Y": np.zeros((1, 1))}, background_load=False), "/v2/models/m/infer", json=X)
    assert r.status_code == 500 and "signature declares [-1]" in r.json()["error"]
