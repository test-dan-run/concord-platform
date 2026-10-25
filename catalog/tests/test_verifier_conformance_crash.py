"""A conformance check that raises something other than an assertion or an HTTP error is one failed line: the
rest of the checks and the capability probe still run, so the stored report keeps every line."""

import types

import httpx
from test_verifier_probe_crash import META, no_outputs

from catalog.verifier import probes


def verify(handler=no_outputs):
    return probes.verify(META, "http://fake", httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler)))


def test_a_check_that_raises_value_error_is_a_failed_line(monkeypatch):
    def crashes(http, meta):
        raise ValueError("boom")

    def passes(http, meta):
        pass

    monkeypatch.setattr(probes, "_conformance", lambda: types.SimpleNamespace(checks_for=lambda meta: [crashes, passes]))
    rep = verify()
    lines = [(x.name, x.status, x.reason) for x in rep.lines]
    assert ("contract: crashes", "fail", "ValueError: boom") in lines
    assert not any(n == "contract: passes" for n, _, _ in lines)
    assert any(n == "segments" for n, _, _ in lines), lines  # the probe still ran
    assert lines[-1][:2] == ("result", "fail") and not rep.conformance_ok and not rep.ok


def test_a_reply_that_is_not_json_is_a_failed_line():
    def not_json(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/v2":
            return httpx.Response(200, content=b"<html>not json</html>")
        return no_outputs(r)

    rep = verify(not_json)
    bad = [x for x in rep.lines if x.name == "contract: oip_server_metadata"]
    assert len(bad) == 1 and bad[0].status == "fail" and bad[0].reason.startswith("JSONDecodeError: "), rep.lines
    before = {x.name for x in verify().lines}  # the same service with a JSON /v2: the other checks and the probe
    assert {x.name for x in rep.lines} == {*before, "contract: oip_server_metadata"} and rep.lines[-1].name == "result"
