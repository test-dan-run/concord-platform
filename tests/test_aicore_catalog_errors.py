"""aicore.catalog.Catalog on error bodies of every JSON shape (a 4xx or 5xx is always a CatalogError)."""

import json

import httpx
import pytest

from aicore.catalog import Catalog, CatalogError


def answering(status: int, body: object) -> httpx.Client:
    """An httpx client whose every request gets `status` and `body` as JSON."""
    content = json.dumps(body).encode()
    return httpx.Client(base_url="http://t", transport=httpx.MockTransport(
        lambda r: httpx.Response(status, content=content, headers={"content-type": "application/json"})))


def catalog_error(status: int, body: object) -> CatalogError:
    with pytest.raises(CatalogError) as exc:
        Catalog("http://t", http=answering(status, body)).health()
    return exc.value


def test_catalog_error_bodies():
    exc = catalog_error(500, [1, 2])  # JSON but not an object: the body is the message
    assert exc.status == 500 and exc.message == "[1, 2]"
    assert Catalog("http://t", http=answering(500, [1, 2])).available() is False  # tools/stack.catalog() falls back
    assert catalog_error(404, {"detail": "gone"}).message == "gone"
    assert catalog_error(502, {"error": "plain"}).message == "plain"
    assert catalog_error(502, {"error": {"message": "upstream down", "type": "upstream"}}).message == "upstream down"
