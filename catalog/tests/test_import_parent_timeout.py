"""A catalog that stops answering while the importer sets "derived from" has already registered every model: the
import says so, and says to import again to set the parents, instead of "not all registered"."""

import json

import httpx
import pytest

from catalog import importer


def test_a_timeout_while_setting_parents_says_every_model_was_registered(tmp_path, monkeypatch, capsys):
    root = tmp_path / "bundle"
    for mid, parent in (("base-model", None), ("child-model", "base-model")):
        (root / "models" / mid).mkdir(parents=True)
        (root / "models" / mid / "registration.json").write_text(
            json.dumps({"registered": True, "tags": [], "parent_id": parent}))
    manifest = {"models": [{"id": "base-model"}, {"id": "child-model"}]}

    def catalog(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/api/health" or r.url.path.endswith("/register"):
            return httpx.Response(200, json={"ok": True})
        raise httpx.ReadTimeout("timed out", request=r)  # the PATCH that sets derived from

    real = httpx.Client
    monkeypatch.setattr(importer.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(catalog), **kw))
    with pytest.raises(importer.BundleError, match="every model it accepted was registered") as exc:
        importer.register(root, manifest, ["base-model", "child-model"], "http://catalog.test", tmp_path / "live.db")
    assert "not all registered" not in str(exc.value)
    assert "registered 2 of 2" in capsys.readouterr().out
