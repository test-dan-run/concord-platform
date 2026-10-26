"""A catalog that answered /api/health and then stops answering partway through registration: the import stops and
says to import again. Its database is not written behind its back, so a model it refused stays unregistered."""

import json

import httpx
import pytest

from catalog import importer
from catalog.db import DB


def test_a_timeout_after_the_health_check_stops_without_writing_the_database(tmp_path, monkeypatch, capsys):
    root = tmp_path / "bundle"
    for mid in ("big-model", "small-model"):
        (root / "models" / mid).mkdir(parents=True)
        (root / "models" / mid / "registration.json").write_text(json.dumps({"registered": True, "tags": []}))
    manifest = {"models": [{"id": "big-model"}, {"id": "small-model"}]}

    def catalog(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/api/health":
            return httpx.Response(200, json={"ok": True})
        if r.url.path == "/api/models/big-model/register":
            return httpx.Response(422, json={"error": {"message": "big-model needs more GPU memory than this host has"}})
        raise httpx.ReadTimeout("timed out", request=r)

    real = httpx.Client
    monkeypatch.setattr(importer.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(catalog), **kw))
    with pytest.raises(importer.BundleError, match="timed out.*imported but not all registered, so import again"):
        importer.register(root, manifest, ["big-model", "small-model"], "http://catalog.test", tmp_path / "live.db")
    out = capsys.readouterr().out
    assert "big-model: the catalog refused to register it (422)" in out and "writing registrations" not in out
    if (tmp_path / "live.db").exists():
        db = DB(tmp_path / "live.db")
        try:
            assert db.all("SELECT model_id FROM registration") == []
        finally:
            db.close()
