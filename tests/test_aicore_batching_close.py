"""Catalog clients can be closed (and used with `with`), and batch_size closes the one it makes."""

import httpx
import pytest

from aicore import batching
from aicore.catalog import Catalog, FakeCatalog


@pytest.fixture
def clients(monkeypatch):
    """Every httpx.Client made during the test, answering `answer[0]` (a status and a body) to each request."""
    made, answer, real = [], [(200, {"recommendation": {"batch": 12}})], httpx.Client

    def client(**kw):
        c = real(**kw, transport=httpx.MockTransport(lambda r: httpx.Response(answer[0][0], json=answer[0][1])))
        made.append(c)
        return c

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("AICORE_CATALOG", "http://catalog.test")
    batching._recommended.cache_clear()
    yield made, answer
    batching._recommended.cache_clear()


def test_catalog_close_and_with():
    with Catalog("http://t") as cat:
        assert not cat._http.is_closed
    assert cat._http.is_closed
    other = Catalog("http://t")
    other.close()
    assert other._http.is_closed
    with FakeCatalog() as fake:  # the fake takes the same calls, and has nothing to close
        fake.close()
    assert fake.health() == {"ok": True}


def test_batch_size_closes_the_catalog_client_it_makes(clients):
    made, answer = clients
    assert batching.batch_size("wavlm-sv", item_s=8.0) == 12
    assert len(made) == 1 and made[0].is_closed
    answer[0] = (500, {"detail": "boom"})
    assert batching.batch_size("bge-m3", item_tokens=512, default=5) == 5  # a catalog error: the default
    assert len(made) == 2 and made[1].is_closed
    assert batching.batch_size("wavlm-sv", item_s=8.0) == 12 and len(made) == 2  # cached: no new client
