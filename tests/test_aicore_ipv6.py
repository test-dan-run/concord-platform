"""aicore service URLs for an IPv6 AI_HOST: the address goes in brackets, so the port stays a port."""

import httpx
import pytest

from aicore.endpoints import endpoint, resolve_base

ENTRY = {"name": "qwen3-8b", "host_port": 8201, "internal_port": 8000}


@pytest.mark.parametrize(("host", "url"), [
    ("::1", "http://[::1]:8201"),  # local: the published port
    ("fd00::5", "http://[fd00::5]:8000"),  # anything else: the internal port
    ("[fd00::5]", "http://[fd00::5]:8000"),  # already in brackets
    ("127.0.0.1", "http://127.0.0.1:8201"),
    ("gpu-box", "http://gpu-box:8000"),
    ("{name}", "http://qwen3-8b:8000"),
])
def test_resolve_base_brackets_ipv6_literals(host, url):
    assert resolve_base(ENTRY, host) == url
    parsed = httpx.URL(url)
    assert parsed.port == (8201 if host in ("::1", "127.0.0.1") else 8000)


def test_ai_host_ipv6_loopback(tmp_path, monkeypatch):
    f = tmp_path / "endpoints.yaml"
    f.write_text("services:\n  default-chat: {name: qwen3-8b, host_port: 8201, internal_port: 8000, "
                 "contract: openai, base_path: /v1}\n")
    monkeypatch.setenv("AICORE_ENDPOINTS", str(f))
    monkeypatch.delenv("AICORE_CATALOG", raising=False)
    monkeypatch.setenv("AI_HOST", "::1")
    ep = endpoint("default-chat")
    assert ep["url"] == "http://[::1]:8201" and ep["base_url"] == "http://[::1]:8201/v1"
    assert httpx.URL(ep["base_url"]).host == "::1"
