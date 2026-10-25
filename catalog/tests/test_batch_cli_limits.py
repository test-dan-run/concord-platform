"""`just batch` (tools/batch.py) shows the limits a custom GPU image starts with, the override and the running ones."""

from __future__ import annotations

from fastapi.testclient import TestClient

from catalog.app import create_app
from catalog.batching import Profile

TRUE = (1.2, 0.004, 0.0003)  # base GiB, a per item-second, b per item-second²


def test_just_batch_prints_the_limits_and_the_override(cat, monkeypatch, capsys):
    import aicore.catalog
    from tools import batch as cli

    cat.profiler._save("wavlm-sv", "bk1", "measured",
                       Profile(kind="audio", base=TRUE[0], a=TRUE[1], b=TRUE[2], max_item_s=20.0, method="torch"), None)
    cat.profiler.set_override("wavlm-sv", 16, None)
    monkeypatch.setattr(cat.builds, "current", lambda p, s, c: type("B", (), {"build_key": "bk1"})())
    with TestClient(create_app(cat, start=False)) as api:
        monkeypatch.setattr(cli, "Catalog", lambda timeout: aicore.catalog.Catalog("http://testserver", http=api))
        assert cli.main(["show", "wavlm-sv", "--item-s", "20"]) == 0
    out = capsys.readouterr().out
    assert "  next start (" in out and "max batch 16 (override)" in out and "s (measured)" in out
    assert "  override: max batch 16, max length from the profile (set " in out


def test_batch_show_prints_what_the_running_service_started_with(capsys):
    from tools import batch as cli

    view = {"model": "wavlm-sv", "status": "none", "profile": None,
            "limits": {"vram_gb": 29.76, "max_batch": None, "max_len_s": None, "measured": {},
                       "source": {"max_batch": "default", "max_len_s": "default"}},
            "override": None, "running": {"vram_gb": 29.0, "started_with": {"max_batch": 128, "max_len_s": 600.0}}}
    cli.show(view)
    out = capsys.readouterr().out
    assert "  next start (29.76 GB): max batch image default, max length image default" in out
    assert "  running: started with max batch 128, max length 600 s" in out
    cli.show({**view, "running": {"vram_gb": None, "started_with": None}})
    assert "  running: started outside the catalog, which did not set its limits" in capsys.readouterr().out
