"""The importer registers through the running catalog, and writes its database only when no catalog answers."""

import json

import httpx
import pytest

from catalog import importer
from catalog.db import DB


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    for mid, reg in (("big-model", {"registered": True, "display_name": "Big", "tags": ["x"]}),
                     ("small-model", {"registered": True, "display_name": "Small", "tags": [], "parent_id": "base-model"})):
        (root / "models" / mid).mkdir(parents=True)
        (root / "models" / mid / "registration.json").write_text(json.dumps(reg))
    return root, {"models": [{"id": "big-model"}, {"id": "small-model"}]}


def with_catalog(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(importer.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def registrations(path):
    if not path.exists():
        return []
    db = DB(path)
    try:
        return [r["model_id"] for r in db.all("SELECT model_id FROM registration ORDER BY model_id")]
    finally:
        db.close()


def test_a_refusal_from_the_catalog_is_shown_and_its_database_is_left_alone(bundle, tmp_path, monkeypatch, capsys):
    root, manifest = bundle
    seen = []

    def catalog(r: httpx.Request) -> httpx.Response:
        seen.append((r.method, r.url.path))
        if r.url.path == "/api/health":
            return httpx.Response(200, json={"ok": True})
        if r.url.path == "/api/models/big-model/register":
            return httpx.Response(422, json={"error": {"message": "big-model needs more GPU memory than this host has"}})
        if r.method == "PATCH":
            return httpx.Response(404, json={"error": {"message": "unknown service 'base-model'"}})
        return httpx.Response(200, json={})

    with_catalog(monkeypatch, catalog)
    importer.register(root, manifest, ["big-model", "small-model"], "http://catalog.test", tmp_path / "live.db")
    out = capsys.readouterr().out
    assert "big-model: the catalog refused to register it (422): big-model needs more GPU memory than this host has" in out
    assert "small-model: the catalog did not set derived from base-model (404): unknown service 'base-model'" in out
    assert "not answering" not in out and registrations(tmp_path / "live.db") == []
    assert ("POST", "/api/models/small-model/register") in seen and ("PATCH", "/api/models/small-model") in seen


def test_a_catalog_that_does_not_answer_gets_the_registrations_in_its_database(bundle, tmp_path, monkeypatch, capsys):
    root, manifest = bundle

    def down(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=r)

    with_catalog(monkeypatch, down)
    importer.register(root, manifest, ["big-model", "small-model"], "http://catalog.test", tmp_path / "live.db")
    assert "is not answering" in capsys.readouterr().out
    assert registrations(tmp_path / "live.db") == ["big-model", "small-model"]


def test_a_failing_health_check_stops_without_writing_the_database(bundle, tmp_path, monkeypatch):
    root, manifest = bundle
    with_catalog(monkeypatch, lambda r: httpx.Response(503, text="starting"))
    with pytest.raises(importer.BundleError, match="answered 503"):
        importer.register(root, manifest, ["big-model"], "http://catalog.test", tmp_path / "live.db")
    assert registrations(tmp_path / "live.db") == []
