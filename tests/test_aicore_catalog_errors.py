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


@pytest.mark.parametrize("body", [[1, 2], "oops", 3, None])
def test_catalog_error_body_that_is_json_but_not_an_object(body):
    exc = catalog_error(500, body)
    assert exc.status == 500 and exc.message == json.dumps(body)
    assert Catalog("http://t", http=answering(500, body)).available() is False  # tools/stack.catalog() falls back


def test_catalog_error_bodies_that_are_objects():
    assert catalog_error(404, {"detail": "gone"}).message == "gone"
    assert catalog_error(502, {"error": "plain"}).message == "plain"
    assert catalog_error(502, {"error": {"message": "upstream down", "type": "upstream"}}).message == "upstream down"
