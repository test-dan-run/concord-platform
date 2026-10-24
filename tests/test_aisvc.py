"""aisvc factories: every OpenAI API type passes the shared conformance checks; OIP validation edges."""

import threading

import numpy as np
import pytest
from aisvc import create_oip_app, create_openai_app, create_openai_inspired_app
from conformance import checks_for, running
from factories import toy_itn_app
from fastapi.testclient import TestClient


def meta(name, contract, api, signature=None):
    m = {"name": name, "contract": contract, "api": api, "health_path": "/health"}
    if signature:
        m["signature"] = signature
    return m


OPENAI_HANDLERS = {
    "chat.completions": lambda req: f"echo: {req.messages[-1].content}",
    "completions": lambda req: " there was a test.",
    "embeddings": lambda req: [[float(len(t)), 1.0, 0.5] for t in req.texts],
    "audio.transcriptions": lambda req: {"text": f"{len(req.file)} bytes", "language": "en"},
    "audio.speech": lambda req: (np.zeros(2400, np.float32), 24000),
}


@pytest.mark.parametrize("api", sorted(OPENAI_HANDLERS))
def test_openai_factory_passes_conformance(api):
    m = meta("svc", "openai", api)
    with running(create_openai_app("svc", api, OPENAI_HANDLERS[api])) as c:
        for check in checks_for(m):
            check(c, m)


def test_transcription_formats():
    with running(create_openai_app("asr", "audio.transcriptions", OPENAI_HANDLERS["audio.transcriptions"])) as c:
        files = {"file": ("a.wav", b"1234", "audio/wav")}
        assert c.post("/v1/audio/transcriptions", data={"model": "asr", "response_format": "text"}, files=files).text == "4 bytes"
        verbose = c.post("/v1/audio/transcriptions", data={"model": "asr", "response_format": "verbose_json"}, files=files).json()
        assert verbose == {"text": "4 bytes", "language": "en"}


def test_openai_streaming_is_rejected_cleanly():
    with running(create_openai_app("llm", "chat.completions", OPENAI_HANDLERS["chat.completions"])) as c:
        r = c.post("/v1/chat/completions", json={"model": "llm", "messages": [{"role": "user", "content": "x"}], "stream": True})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_unknown_route_is_openai_shaped():
    with running(create_openai_app("llm", "chat.completions", OPENAI_HANDLERS["chat.completions"])) as c:
        r = c.get("/v1/nope")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_handler_crash_is_500_with_message():
    def boom(req):
        raise RuntimeError("kaput")

    with running(create_openai_app("llm", "chat.completions", boom)) as c:
        r = c.post("/v1/chat/completions", json={"model": "llm", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 500 and "kaput" in r.json()["error"]["message"]


def test_failed_load_keeps_health_503():
    def bad_load():
        raise FileNotFoundError("weights missing")

    app = create_openai_inspired_app("n", "text.transform", lambda t, o: t, load=bad_load)
    with TestClient(app) as c:
        assert not app.state.svc.wait(5)
        r = c.get("/health")
        assert r.status_code == 503 and "weights missing" in r.json()["error"]


def test_unknown_inspired_api_type_rejected():
    with pytest.raises(ValueError, match="define it in contracts/README.md"):
        create_openai_inspired_app("n", "image.resize", lambda t, o: t)


# ----------------------------------------------------------------------------- text.itn

def itn(c, input, **options):
    return c.post("/v1/text/itn", json={"model": "itn-toy", "input": input, "options": options})


def test_itn_factory_passes_conformance():
    m = meta("itn-toy", "openai-inspired", "text.itn")
    with running(toy_itn_app()) as c:
        for check in checks_for(m):
            check(c, m)


def test_itn_spans_only_when_asked():
    with running(toy_itn_app()) as c:
        plain = itn(c, "meet at three").json()["data"][0]
        assert plain == {"object": "text.itn", "index": 0, "output": "meet at 3"}
        item = itn(c, "meet at three", return_spans=True).json()["data"][0]
        assert item["spans"] == [{"input_start": 8, "input_end": 13, "output_start": 8, "output_end": 9, "class": "CARDINAL"}]
        assert itn(c, "three", classes="CARDINAL", return_spans=True).json()["data"][0]["output"] == "3"


@pytest.mark.parametrize("options,fragment", [
    ({"language": "fr-FR"}, "unsupported language"),
    ({"classes": "TIME"}, "unsupported classes"),
    ({"nope": 1}, "options.nope"),
])
def test_itn_unsupported_options_are_invalid_options(options, fragment):
    with running(toy_itn_app()) as c:
        r = itn(c, "three", **options)
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options", r.text
        assert fragment in r.json()["error"]["message"]


def test_itn_limits_are_invalid_request():
    with running(toy_itn_app()) as c:
        for input in (["x"] * 257, "x" * 10_001):
            r = itn(c, input)
            assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request", r.text


def test_itn_invalid_handler_span_is_500():
    def bad(texts, options):
        return [(t, [{"input_start": 0, "input_end": len(t) + 5, "output_start": 0, "output_end": 0, "class": "X"}]) for t in texts]

    with running(toy_itn_app(bad)) as c:
        r = itn(c, "three")
        assert r.status_code == 500 and "invalid span" in r.json()["error"]["message"]


def test_itn_options_model_must_subclass_itn_options():
    from pydantic import BaseModel

    class Other(BaseModel):
        language: str | None = None

    with pytest.raises(ValueError, match="subclass aisvc.ItnOptions"):
        create_openai_inspired_app("n", "text.itn", lambda t, o: [], options_model=Other)


# ----------------------------------------------------------------------------- OIP

SIG = {"inputs": [{"name": "A", "datatype": "INT16", "shape": [2, -1]}],
       "outputs": [{"name": "B", "datatype": "FP32", "shape": [-1]}]}


@pytest.fixture(scope="module")
def oip():
    app = create_oip_app("m", SIG, lambda x, p: {"B": x["A"].sum(axis=0)}, version="3")
    with running(app) as c:
        yield c


def infer(c, inputs, path="/v2/models/m/infer", **extra):
    return c.post(path, json={"inputs": inputs, **extra})


def a(shape, data, datatype="INT16", name="A"):
    return {"name": name, "datatype": datatype, "shape": shape, "data": data}


def test_oip_flat_and_nested_data(oip):
    flat = infer(oip, [a([2, 2], [1, 2, 3, 4])]).json()
    nested = infer(oip, [a([2, 2], [[1, 2], [3, 4]])]).json()
    assert flat["outputs"] == nested["outputs"] == [{"name": "B", "datatype": "FP32", "shape": [2], "data": [4.0, 6.0]}]
    assert flat["model_version"] == "3"


@pytest.mark.parametrize("inputs,fragment", [
    ([a([3, 2], [0] * 6)], "shape [3, 2] does not match signature [2, -1]"),
    ([a([2], [0, 0])], "does not match signature"),
    ([a([2, 2], [0, 0, 0])], "3 values do not fill shape [2, 2]"),
    ([a([2, 1], [0.5, 1.5])], "INT16 data must be integers"),
    ([a([2, 1], ["x", "y"])], "INT16 data must be numeric"),
    ([a([2, 1], [0, 0], datatype="FP32")], "datatype FP32 does not match signature INT16"),
    ([a([2, 1], [0, 0]), a([1], [0], name="Z")], "unexpected input 'Z'"),
    ([a([2, 1], [0, 0]), a([2, 1], [0, 0])], "given twice"),
])
def test_oip_validation_errors_are_400(oip, inputs, fragment):
    r = infer(oip, inputs)
    assert r.status_code == 400 and fragment in r.json()["error"], r.text


def test_oip_missing_input_and_unknown_output(oip):
    two = {"inputs": [{"name": "A", "datatype": "INT16", "shape": [2, 1], "data": [1, 2]}],
           "outputs": [{"name": "NOPE"}]}
    r = oip.post("/v2/models/m/infer", json=two)
    assert r.status_code == 400 and "unknown outputs" in r.json()["error"]
    sig2 = {"inputs": SIG["inputs"] + [{"name": "C", "datatype": "FP32", "shape": [1]}], "outputs": SIG["outputs"]}
    with running(create_oip_app("m2", sig2, lambda x, p: {"B": np.zeros(1)})) as c:
        r = infer(c, [a([2, 1], [1, 2])], path="/v2/models/m2/infer")
        assert r.status_code == 400 and "missing inputs ['C']" in r.json()["error"]


def test_oip_versions_and_unknown_model(oip):
    assert oip.get("/v2/models/m/versions/3").json()["versions"] == ["3"]
    assert infer(oip, [a([2, 1], [1, 2])], path="/v2/models/m/versions/3/infer").status_code == 200
    assert oip.get("/v2/models/m/versions/9").status_code == 404
    r = oip.get("/v2/models/other")
    assert r.status_code == 404 and "not found" in r.json()["error"]
    assert oip.post("/v2/models/m/infer", content=b"not json", headers={"content-type": "application/json"}).status_code == 400


def test_oip_bad_handler_output_is_500():
    with running(create_oip_app("m", SIG, lambda x, p: {"B": np.zeros((2, 2))})) as c:
        r = infer(c, [a([2, 1], [1, 2])])
        assert r.status_code == 500 and "signature declares [-1]" in r.json()["error"]


def test_oip_not_ready_is_503():
    gate = threading.Event()
    app = create_oip_app("m", SIG, lambda x, p: {"B": np.zeros(1)}, load=lambda: gate.wait(5))
    with TestClient(app) as c:
        assert c.get("/v2/health/ready").status_code == 503
        assert c.get("/v2/models/m/ready").json() == {"name": "m", "ready": False}
        assert c.get("/v2/health/live").json() == {"live": True}
        r = infer(c, [a([2, 1], [1, 2])])
        assert r.status_code == 503 and "not ready" in r.json()["error"]
        gate.set()
        assert app.state.svc.wait(5)
        assert c.get("/v2/health/ready").json() == {"ready": True}
