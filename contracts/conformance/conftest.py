"""Live conformance runs: pytest contracts/conformance [--service NAME ...] [--base-url URL].

Services come from the generated endpoints map (via aicore, so AI_HOST is honoured). A service
that is not reachable, or still loading, is skipped with a message rather than failed.
"""

from __future__ import annotations

import httpx
import pytest

from conformance.checks import checks_for


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("conformance")
    group.addoption("--service", action="append", default=[], help="service name or alias (repeatable; default: all)")
    group.addoption("--base-url", default=None, help="override the service root URL (single --service only)")


def _selected(config: pytest.Config) -> list[dict]:
    from aicore import endpoint
    from aicore.endpoints import AicoreError, load_endpoints

    try:
        services = load_endpoints()
    except AicoreError as exc:
        raise pytest.UsageError(str(exc)) from None
    names = config.getoption("--service") or sorted({e["name"] for e in services.values()})
    try:
        entries = [endpoint(n) for n in names]
    except KeyError as exc:
        raise pytest.UsageError(str(exc)) from None
    base_url = config.getoption("--base-url")
    if base_url:
        if len(entries) != 1:
            raise pytest.UsageError("--base-url needs exactly one --service")
        entries[0]["url"] = base_url.rstrip("/")
    return entries


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if {"svc", "check"} <= set(metafunc.fixturenames):
        params, ids = [], []
        for entry in _selected(metafunc.config):
            for check in checks_for(entry):
                params.append((entry, check))
                ids.append(f"{entry['name']}::{check.__name__}")
        metafunc.parametrize(("svc", "check"), params, ids=ids)


_clients: dict[str, httpx.Client] = {}
_skip_reason: dict[str, str | None] = {}


@pytest.fixture
def live_client(svc: dict) -> httpx.Client:
    name, url = svc["name"], svc["url"]
    if name not in _skip_reason:
        client = httpx.Client(base_url=url, timeout=60)
        try:
            r = client.get(svc["health_path"], timeout=5)
            _skip_reason[name] = (None if r.status_code == 200 else
                                  f"{name} at {url} is reachable but not ready ({svc['health_path']} -> {r.status_code}); "
                                  "wait for it to finish loading")
        except httpx.TransportError as exc:
            _skip_reason[name] = (f"{name} not reachable at {url} ({type(exc).__name__}); "
                                  f"start it with `just up {name}`")
        _clients[name] = client
    if _skip_reason[name]:
        pytest.skip(_skip_reason[name])
    return _clients[name]


def pytest_sessionfinish(session: pytest.Session) -> None:
    for c in _clients.values():
        c.close()
