"""Every catalog API route is in catalog/README.md, and the API docs are at /api/docs only (no /redoc)."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

README = Path(__file__).resolve().parents[1] / "README.md"


def _params(path: str) -> str:
    return re.sub(r"\{[^}/]*\}", "{}", path)  # {id} in the README, {model_id} in the code


@pytest.fixture
def app(cat):
    return create_app(cat, start=False)


def test_every_api_route_is_in_the_readme(app):
    text = _params(README.read_text(encoding="utf-8"))
    paths = sorted({_params(r.path) for r in app.routes if getattr(r, "path", "").startswith("/api/")})
    missing = [p for p in paths if not re.search(re.escape(p) + r"(?![\w/{])", text)]
    assert missing == []


def test_the_api_docs_are_at_api_docs_only(app):
    with TestClient(app) as c:
        docs = c.get("/api/docs")
        assert docs.status_code == 200 and "swagger" in docs.text.lower()
        assert "/api/index/stats" in c.get("/api/openapi.json").json()["paths"]
        redoc = c.get("/redoc")
        assert "redoc" not in redoc.text.lower() and redoc.text.startswith("The catalog UI is not built")
