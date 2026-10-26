"""aisvc.batch: POST /platform/batch-limits takes whole numbers only, so a float or a bool cannot reach a batcher."""

import pytest
from aisvc import batch
from aisvc._core import create_base_app
from fastapi.testclient import TestClient

# a whole-valued float, a bool (an int subclass), and a bad second value after a good first one
BAD = [{"max_items": 4.0}, {"max_items": True}, {"max_items": 8, "max_len": [100]}]


@pytest.fixture
def limits():
    mb = batch.MicroBatcher(lambda xs: xs, length=lambda x: 1, limits=batch.Limits(4, 100), name="types-test")
    yield mb.limits
    batch._LIMITS.pop("types-test", None)


@pytest.mark.parametrize("body", BAD, ids=str)
def test_not_a_whole_number_is_422_and_changes_nothing(limits, body):
    with TestClient(create_base_app("t", title="t", load=None, background=False)) as c:
        r = c.post("/platform/batch-limits", json=body)
    assert r.status_code == 422, r.text
    assert set(r.json()) == {"error"} and "batch limits" in r.json()["error"]
    assert (limits.max_items, limits.max_len) == (4, 100)
