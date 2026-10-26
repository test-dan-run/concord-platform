"""batch_size falls back to the default on any odd recommendation; TextServiceClient gives a plain error string."""

import httpx
import pytest

from aicore import batching
from aicore.text import ServiceError, TextServiceClient


@pytest.fixture
def catalog_answers(monkeypatch):
    """The catalog's batch view for every request; `answer[0]` is the JSON body."""
    answer, real = [None], httpx.Client

    def client(**kw):
        return real(**kw, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=answer[0])))

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("AICORE_CATALOG", "http://catalog.test")
    batching._recommended.cache_clear()
    yield answer
    batching._recommended.cache_clear()


@pytest.mark.parametrize("view", [{"recommendation": {"batch": "many"}}, {"recommendation": ["8"]},
                                  {"recommendation": "8"}, {"recommendation": {"batch": [8]}}],
                         ids=["word", "list", "string", "list-batch"])
def test_an_odd_recommendation_gives_the_default(catalog_answers, view):
    catalog_answers[0] = view
    assert batching.batch_size("wavlm-sv", item_s=8.0, default=5) == 5


def test_a_good_recommendation_still_counts(catalog_answers):
    catalog_answers[0] = {"recommendation": {"batch": 12}}
    assert batching.batch_size("wavlm-sv", item_s=8.0) == 12


def service_error(status: int, content: bytes) -> ServiceError:
    http = httpx.Client(base_url="http://t", transport=httpx.MockTransport(
        lambda r: httpx.Response(status, content=content, headers={"content-type": "application/json"})))
    with pytest.raises(ServiceError) as exc:
        TextServiceClient("http://t", "text-normalizer", http=http).transform("x")
    return exc.value


def test_a_plain_error_string_is_the_message():
    exc = service_error(503, b'{"error": "model is loading"}')
    assert exc.status == 503 and exc.message == "model is loading" and exc.type is None
    assert str(exc) == "503 error: model is loading"


def test_an_error_object_and_other_bodies():
    exc = service_error(400, b'{"error": {"message": "bad option", "type": "invalid_request_error", "code": "x"}}')
    assert (exc.message, exc.type, exc.code) == ("bad option", "invalid_request_error", "x")
    assert service_error(500, b"[1, 2]").message == "[1, 2]"
    assert service_error(500, b'{"error": 3}').message == '{"error": 3}'
    assert service_error(502, b"bad gateway").message == "bad gateway"
