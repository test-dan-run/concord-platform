"""stack.py treats only a refused connection as "no catalog": a catalog that fails or times out is an error."""

import httpx
import pytest
from aicore.catalog import Catalog, CatalogError, FakeCatalog

from tools import stack
from tools.schema import load_platform


@pytest.fixture
def docker(monkeypatch, tmp_path):
    """Docker mocked out, so the direct path (if taken) records its compose calls instead of running them."""
    calls = []
    monkeypatch.delenv("AI_STACK_DIRECT", raising=False)
    monkeypatch.setattr(stack, "LEASES_DIR", tmp_path / "run" / "leases")
    monkeypatch.setattr(stack, "prepare", load_platform)
    monkeypatch.setattr(stack, "require_docker", lambda: None)
    monkeypatch.setattr(stack, "ensure_network", lambda name: None)
    monkeypatch.setattr(stack, "ensure_assets", lambda p, s: None)
    monkeypatch.setattr(stack.images, "image_exists", lambda ref: True)
    monkeypatch.setattr(stack, "measure_gpu", lambda: None)
    monkeypatch.setattr(stack, "containers", lambda p: {})
    monkeypatch.setattr(stack, "compose", lambda p, *args, **kw: calls.append(args))
    monkeypatch.setattr(stack, "wait_healthy", lambda p, name, timeout: None)
    return calls


def raising(exc):
    def health(self):
        raise exc
    return health


@pytest.mark.parametrize("exc", [CatalogError(503, "busy"), httpx.ReadTimeout("timed out")], ids=["503", "timeout"])
def test_a_catalog_that_does_not_answer_stops_the_command(docker, monkeypatch, capsys, exc):
    monkeypatch.setattr(Catalog, "health", raising(exc))
    assert stack.main(["up", "default-vad"]) == 1
    err = capsys.readouterr().err
    assert "error: catalog at " in err and "is not answering" in err and "AI_STACK_DIRECT=1 bypasses it" in err
    assert docker == []


def test_a_refused_connection_means_no_catalog(docker, monkeypatch):
    monkeypatch.setattr(Catalog, "health", raising(httpx.ConnectError("refused")))
    assert stack.catalog() is None
    assert stack.main(["up", "default-vad"]) == 0
    assert [a[-1] for a in docker if a[0] == "up"] == ["silero-vad"]


def test_a_transport_error_mid_command_is_an_error_not_a_traceback(monkeypatch, capsys):
    fake = FakeCatalog([{"id": "silero-vad", "state": "stopped", "leased_by": [], "state_detail": None}])

    def start(model_id):
        raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr(fake, "start", start)
    monkeypatch.setattr(stack, "catalog", lambda: fake)
    monkeypatch.setattr(stack, "prepare", load_platform)
    assert stack.main(["up", "default-vad"]) == 1
    assert "error: catalog at fake://catalog is not answering (server disconnected)" in capsys.readouterr().err
