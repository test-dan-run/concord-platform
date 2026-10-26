"""With the catalog running, `stack.py status` shows its lease holders and `lease --ttl` says the TTL is ignored."""

import time

import pytest
from aicore.catalog import FakeCatalog

from tools import stack
from tools.leases import Lease, Leases
from tools.schema import load_platform


@pytest.fixture
def fake(monkeypatch, tmp_path):
    fake = FakeCatalog([{"id": "whisper-large-v3", "state": "running", "leased_by": ["bench:job-1"], "state_detail": None},
                        {"id": "silero-vad", "state": "stopped", "leased_by": [], "state_detail": None}])
    monkeypatch.setattr(stack, "catalog", lambda: fake)
    monkeypatch.setattr(stack, "LEASES_DIR", tmp_path / "run" / "leases")
    monkeypatch.setattr(stack, "require_docker", lambda: None)
    monkeypatch.setattr(stack, "prepare", load_platform)
    monkeypatch.setattr(stack, "containers", lambda p: {
        "whisper-large-v3": stack.Container("id", "whisper-large-v3", "running", "healthy", True)})
    return fake


def leased_by(out: str, name: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(name + " ")).split()[-1]


def test_status_shows_the_catalogs_lease_holders(fake, capsys):
    Leases(stack.LEASES_DIR).write(Lease("old:job", ["silero-vad"], time.time() + 60))  # not the catalog's
    assert stack.main(["status"]) == 0
    out = capsys.readouterr().out
    assert leased_by(out, "whisper-large-v3") == "bench:job-1"
    assert leased_by(out, "silero-vad") == "-"


def test_status_without_the_catalog_reads_the_lease_files(fake, monkeypatch, capsys):
    monkeypatch.setattr(stack, "catalog", lambda: None)
    Leases(stack.LEASES_DIR).write(Lease("old:job", ["silero-vad"], time.time() + 60))
    assert stack.main(["status"]) == 0
    out = capsys.readouterr().out
    assert leased_by(out, "silero-vad") == "old:job" and leased_by(out, "whisper-large-v3") == "-"


def test_lease_ttl_is_ignored_with_a_note_in_catalog_mode(fake, capsys):
    fake.request_lease("synth:run-1", ["silero-vad"])
    assert stack.main(["lease", "--holder", "synth:run-1", "--ttl", "300"]) == 0
    assert "note: --ttl is ignored" in capsys.readouterr().out
    assert stack.main(["lease", "--holder", "synth:run-1"]) == 0
    assert "--ttl" not in capsys.readouterr().out
