"""aisvc openai-inspired: an error while the factory builds the response from a handler's result is a 500 in the
OpenAI error shape, not a plain-text 500."""

import pytest
from aisvc import create_openai_inspired_app
from fastapi.testclient import TestClient


def post(app, path, body):
    with TestClient(app, raise_server_exceptions=False) as c:
        return c.post(path, headers={"origin": "http://localhost:5173"}, json=body)


@pytest.mark.parametrize("api, result, path", [
    ("text.transform", lambda texts: [None for _ in texts], "/v1/text/transform"),
    ("text.itn", lambda texts: [(None, []) for _ in texts], "/v1/text/itn"),
], ids=["transform-none", "itn-none"])
def test_inspired_result_errors_are_openai_shaped_500s(api, result, path):
    app = create_openai_inspired_app("m", api, lambda texts, options: result(texts), background_load=False)
    r = post(app, path, {"model": "m", "input": ["a", "b"]})
    assert r.status_code == 500 and r.headers["content-type"] == "application/json", r.text
    err = r.json()["error"]
    assert err["type"] == "server_error" and err["message"].startswith("TypeError")
    assert r.headers["access-control-allow-origin"] == "*"  # answered inside the app, so CORS still applies


def test_inspired_results_keep_their_shape():
    app = create_openai_inspired_app("m", "text.transform", lambda texts, options: [t.upper() for t in texts],
                                     background_load=False)
    r = post(app, "/v1/text/transform", {"model": "m", "input": ["ab", "c"]})
    assert r.status_code == 200
    body = r.json()
    assert body["data"] == [{"object": "text.transform", "index": 0, "output": "AB"},
                            {"object": "text.transform", "index": 1, "output": "C"}]
    assert body["object"] == "list" and body["model"] == "m" and body["usage"] == {"input_chars": 3, "output_chars": 3}
    spans = [{"input_start": 0, "input_end": 2, "output_start": 0, "output_end": 1, "class": "CARDINAL"}]
    app = create_openai_inspired_app("m", "text.itn", lambda texts, options: [("2", spans) for _ in texts],
                                     background_load=False)
    r = post(app, "/v1/text/itn", {"model": "m", "input": "tw", "options": {"return_spans": True}})
    assert r.status_code == 200
    assert r.json()["data"] == [{"object": "text.itn", "index": 0, "output": "2", "spans": spans}]
    assert r.json()["usage"] == {"input_chars": 2, "output_chars": 1}
