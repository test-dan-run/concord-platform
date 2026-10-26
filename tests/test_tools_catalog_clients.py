"""stack.py and batch.py close the catalog clients they open; `up --holder --ttl` keeps to the catalog's limit."""

import httpx
import pytest
from aicore.catalog import FakeCatalog

from catalog.app import LeaseIn
from tools import batch, stack
from tools.schema import load_platform


@pytest.fixture
def clients(monkeypatch):
    """Every httpx.Client made during the test; `handler[0]` answers its requests."""
    made, handler, real = [], [None], httpx.Client

    def client(**kw):
        c = real(**kw, transport=httpx.MockTransport(lambda r: handler[0](r)))
        made.append(c)
        return c

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("AICORE_CATALOG", "http://catalog.test")
    monkeypatch.delenv("AI_STACK_DIRECT", raising=False)
    return made, handler


def answer(routes):
    """A handler answering JSON by path; any other path is a 404."""
    def handle(r):
        if r.url.path in routes:
            return httpx.Response(200, json=routes[r.url.path])
        return httpx.Response(404, json={"detail": f"no {r.url.path}"})
    return handle


def refuse(r):
    raise httpx.ConnectError("refused")


def test_catalog_closes_its_client_when_the_health_check_fails(clients):
    made, handler = clients
    handler[0] = refuse  # no catalog listens
    assert stack.catalog() is None
    assert len(made) == 1 and made[0].is_closed
    handler[0] = lambda r: httpx.Response(503, json={"detail": "busy"})  # the catalog fails
    with pytest.raises(stack.StackError, match="is not answering"):
        stack.catalog()
    assert len(made) == 2 and made[1].is_closed


def test_a_catalog_command_closes_the_client(clients, capsys):
    made, handler = clients
    handler[0] = answer({"/api/health": {"ok": True}, "/api/leases": []})
    assert stack.main(["leases"]) == 0
    assert "no leases" in capsys.readouterr().out
    assert len(made) == 1 and made[0].is_closed


def test_status_closes_the_catalog_client(clients, monkeypatch, capsys):
    made, handler = clients
    handler[0] = answer({"/api/health": {"ok": True}, "/api/models": []})
    monkeypatch.setattr(stack, "require_docker", lambda: None)
    monkeypatch.setattr(stack, "prepare", load_platform)
    monkeypatch.setattr(stack, "containers", lambda p: {})
    assert stack.main(["status"]) == 0
    assert len(made) == 1 and made[0].is_closed


@pytest.mark.parametrize("argv,code", [(["profile", "wavlm-sv"], 0), (["show", "no-such-model"], 1)])
def test_just_batch_closes_the_catalog_client(clients, argv, code, capsys):
    made, handler = clients
    handler[0] = answer({"/api/models/wavlm-sv/batch-profile": {"model": "wavlm-sv", "measuring": "pending"}})
    assert batch.main(argv) == code
    assert len(made) == 1 and made[0].is_closed


@pytest.fixture
def fake(monkeypatch):
    fake = FakeCatalog([{"id": "silero-vad", "state": "running", "leased_by": [], "state_detail": None}])
    monkeypatch.setattr(stack, "catalog", lambda: fake)
    monkeypatch.setattr(stack, "prepare", load_platform)
    return fake


def test_up_with_a_ttl_over_the_catalogs_limit_is_refused_up_front(fake, capsys):
    assert stack.main(["up", "silero-vad", "--holder", "synth:run-1", "--ttl", "3601", "--timeout", "5"]) == 1
    err = capsys.readouterr().err
    assert "error: --ttl is at most 3600 seconds with the catalog running" in err
    assert not [c for c in fake.calls if c[0] == "lease"]
    assert stack.main(["up", "silero-vad", "--holder", "synth:run-1", "--ttl", "3600", "--timeout", "5"]) == 0
    assert [c for c in fake.calls if c[0] == "lease"] == [("lease", ("synth:run-1", ("silero-vad",)))]


def test_the_ttl_limit_is_the_catalogs():
    limit = next(m.le for m in LeaseIn.model_fields["ttl_s"].metadata if hasattr(m, "le"))
    assert stack.CATALOG_MAX_TTL == limit
