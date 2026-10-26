"""`stack.py status` only reads, so a catalog that fails or times out does not stop it: it warns and shows
the table with LEASED BY "?" (the catalog's leases are unknown, and run/leases are not the catalog's)."""

import time

import httpx
import pytest
from aicore.catalog import Catalog, CatalogError, FakeCatalog

from tools import stack
from tools.leases import Lease, Leases
from tools.schema import load_platform


@pytest.fixture
def docker(monkeypatch, tmp_path):
    monkeypatch.delenv("AI_STACK_DIRECT", raising=False)
    monkeypatch.setattr(stack, "LEASES_DIR", tmp_path / "run" / "leases")
    monkeypatch.setattr(stack, "require_docker", lambda: None)
    monkeypatch.setattr(stack, "prepare", load_platform)
    monkeypatch.setattr(stack, "containers", lambda p: {
        "silero-vad": stack.Container("id", "silero-vad", "running", "healthy", False)})
    Leases(stack.LEASES_DIR).write(Lease("old:job", ["silero-vad"], time.time() + 60))  # not the catalog's


def leased_by(out: str) -> dict[str, str]:
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("NAME "))
    return {line.split()[0]: line.split()[-1] for line in lines[start + 1:]}


def raising(exc):
    def fail(self, *args, **kwargs):
        raise exc
    return fail


def test_status_shows_the_table_when_the_catalog_health_check_fails(docker, monkeypatch, capsys):
    monkeypatch.setattr(Catalog, "health", raising(httpx.ReadTimeout("timed out")))
    assert stack.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "warning: catalog at " in out and "is not answering" in out
    cols = leased_by(out)
    assert cols["silero-vad"] == "?" and set(cols.values()) == {"?"}
    assert "running" in next(line for line in out.splitlines() if line.startswith("silero-vad "))


def test_status_shows_the_table_when_the_model_list_fails(docker, monkeypatch, capsys):
    fake = FakeCatalog([{"id": "silero-vad", "state": "running", "leased_by": ["bench:job-1"], "state_detail": None}])
    monkeypatch.setattr(fake, "models", raising(CatalogError(503, "busy")).__get__(fake))
    monkeypatch.setattr(stack, "catalog", lambda: fake)
    assert stack.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "warning: catalog" in out
    assert set(leased_by(out).values()) == {"?"}
