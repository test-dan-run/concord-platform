"""aicore.OIPClient error messages: a service's own error string, and the catalog gateway's error object."""

import httpx
import pytest

from aicore.oip import OIPClient, OIPError
from catalog.gateway import error as gateway_error

SIG = {"inputs": [], "outputs": []}


def oip_error(status: int, content: bytes) -> OIPError:
    http = httpx.Client(base_url="http://t", transport=httpx.MockTransport(
        lambda r: httpx.Response(status, content=content, headers={"content-type": "application/json"})))
    with pytest.raises(OIPError) as exc:
        OIPClient("http://t", "silero-vad", SIG, http=http).metadata()
    return exc.value


def test_error_messages_from_the_gateway_the_service_and_other_bodies():
    exc = oip_error(503, gateway_error(503, "silero-vad is starting", "unavailable").body)
    assert exc.status == 503 and exc.message == "silero-vad is starting"
    assert str(exc) == "503: silero-vad is starting"
    assert oip_error(404, b'{"error": "model other not found"}').message == "model other not found"
    assert oip_error(500, b"[1, 2]").message == "[1, 2]"
    assert oip_error(500, b'{"error": {"type": "x"}}').message == '{"error": {"type": "x"}}'  # no message: the body
