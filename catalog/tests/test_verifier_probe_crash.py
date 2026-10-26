"""A probe that crashes, or an answer without the expected output, is one failed line: the rest of the report stays."""

import httpx

from catalog.verifier import probes

SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
                  {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "example": [16000]}],
       "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}]}
META = {"name": "fake-vad", "contract": "oip-v2", "api": "infer", "signature": SIG, "tasks": ["voice-activity-detection"],
        "health_path": "/health"}


def no_outputs(r: httpx.Request) -> httpx.Response:
    """An OIP service that answers every infer with no outputs at all."""
    r.read()
    path = r.url.path
    if path in ("/health", "/v2/health/live", "/v2/health/ready"):
        return httpx.Response(200, json={"live": True, "ready": True})
    if path == "/v2":
        return httpx.Response(200, json={"name": "fake", "version": "1", "extensions": []})
    if path == "/v2/models/fake-vad":
        return httpx.Response(200, json={"name": "fake-vad", "platform": "x", **SIG})
    if path == "/v2/models/fake-vad/ready":
        return httpx.Response(200, json={"name": "fake-vad", "ready": True})
    return httpx.Response(200, json={"model_name": "fake-vad", "outputs": []})


def verify(handler=no_outputs):
    return probes.verify(META, "http://fake", httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler)))


def test_a_segmenter_answer_without_segments_is_a_failed_line():
    rep = verify()
    names = [x.name for x in rep.lines]
    assert ("segments", "fail", "no SEGMENTS output returned") in [(x.name, x.status, x.reason) for x in rep.lines]
    assert names[-1] == "result" and any(n.startswith("contract") for n in names) and not rep.ok, rep.lines


def test_a_probe_that_crashes_keeps_the_conformance_lines(monkeypatch):
    def crash(rep, http, meta):
        rep.tasks.append("vad")  # it detected a task before it crashed: the crash clears it
        raise KeyError("SEGMENTS")

    monkeypatch.setattr(probes, "probe_segmenter", crash)
    rep = verify()
    lines = [(x.name, x.status) for x in rep.lines]
    assert any(n.startswith("contract") for n, _ in lines) and ("probe", "fail") in lines and lines[-1] == ("result", "fail")
    assert "KeyError" in next(x.reason for x in rep.lines if x.name == "probe")
    assert not rep.ok and rep.tasks == []
