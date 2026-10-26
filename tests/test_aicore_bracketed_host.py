"""aicore service URLs for an AI_HOST already written in brackets: [::1] is the local host, like ::1."""

import pytest

from aicore.endpoints import endpoint, resolve_base

ENTRY = {"name": "qwen3-8b", "host_port": 8201, "internal_port": 8000}


@pytest.mark.parametrize(("host", "url"), [
    ("[::1]", "http://[::1]:8201"),  # local: the published port, as for ::1
    ("::1", "http://[::1]:8201"),
    ("[fd00::5]", "http://[fd00::5]:8000"),  # anything else: the internal port
])
def test_bracketed_loopback_gets_the_published_port(host, url):
    assert resolve_base(ENTRY, host) == url


def test_ai_host_bracketed_loopback(tmp_path, monkeypatch):
    f = tmp_path / "endpoints.yaml"
    f.write_text("services:\n  default-chat: {name: qwen3-8b, host_port: 8201, internal_port: 8000, "
                 "contract: openai, base_path: /v1}\n")
    monkeypatch.setenv("AICORE_ENDPOINTS", str(f))
    monkeypatch.delenv("AICORE_CATALOG", raising=False)
    monkeypatch.setenv("AI_HOST", "[::1]")
    assert endpoint("default-chat")["base_url"] == "http://[::1]:8201/v1"
