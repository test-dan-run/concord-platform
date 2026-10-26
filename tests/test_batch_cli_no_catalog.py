"""`just batch` and `just batch-profile` print one error line, not a traceback, when no catalog answers."""

import httpx
import pytest

from tools import batch


def failing(exc):
    def handle(r):
        raise exc
    return handle


@pytest.fixture
def transport(monkeypatch):
    """Every httpx.Client made during the test answers with `handler[0]`."""
    handler, real = [None], httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(**kw, transport=httpx.MockTransport(lambda r: handler[0](r))))
    monkeypatch.setenv("AICORE_CATALOG", "http://catalog.test")
    return handler


@pytest.mark.parametrize("argv", [["show", "silero-vad"], ["profile", "silero-vad"]], ids=["batch", "batch-profile"])
@pytest.mark.parametrize("exc,why", [(httpx.ConnectError("connection refused"), "connection refused"),
                                     (httpx.ReadTimeout(""), "ReadTimeout")], ids=["refused", "timeout"])
def test_no_catalog_is_one_error_line(transport, capsys, argv, exc, why):
    transport[0] = failing(exc)
    assert batch.main(argv) == 1
    err = capsys.readouterr().err
    assert err == f"error: catalog at http://catalog.test is not answering ({why})\n"


def test_nothing_listening_is_one_error_line(monkeypatch, capsys):
    monkeypatch.setenv("AICORE_CATALOG", "http://127.0.0.1:9")  # nothing listens on the discard port
    assert batch.main(["show", "silero-vad"]) == 1
    assert capsys.readouterr().err.startswith("error: catalog at http://127.0.0.1:9 is not answering (")
