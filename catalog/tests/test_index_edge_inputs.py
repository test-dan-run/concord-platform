"""Output index inputs at the edges: a span that is not a finite number is refused, and an output id comes back
in a form that finds the same entry again."""

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app

KEY = {"build_key": "bk", "audio_sha256": "a" * 64, "params_hash": "p" * 16, "span_start": 0.0, "span_end": 1.0}


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


def raw(api, path: str, body: str):
    return api.post(path, content=body, headers={"content-type": "application/json"})


@pytest.mark.parametrize("bad", ["NaN", "1" + "0" * 400], ids=["nan", "1e400"])  # isfinite fails; float() overflows
def test_a_span_that_is_not_a_finite_number_is_refused(api, bad):
    good = '{"build_key": "bk", "audio_sha256": "%s", "params_hash": "p", "span_start": 0, "span_end": 1, ' \
           '"output_id": 1}' % ("a" * 64)
    worse = good.replace('"span_start": 0', f'"span_start": {bad}')
    r = raw(api, "/api/index/record", '{"app": "x", "entries": [%s, %s]}' % (good, worse))
    assert r.status_code == 422, r.text
    assert api.get("/api/index/stats").json() == []  # nothing of the batch was recorded
    key = '{"build_key": "bk", "audio_sha256": "%s", "params_hash": "p", "span_start": 0, "span_end": %s}' % ("a" * 64, bad)
    assert raw(api, "/api/index/lookup", '{"app": "x", "keys": [%s]}' % key).status_code == 422


# The boundaries of the integer form; test_catalog_client covers leading zeros, -0, long and non-ASCII digit ids.
IDS = [("--5", "--5"), (" 5", " 5"), ("9" * 19, int("9" * 19)), (0, 0)]


@pytest.mark.parametrize("oid, back", IDS, ids=[repr(o)[:12] for o, _ in IDS])
def test_an_output_id_comes_back_in_a_form_that_finds_it_again(api, oid, back):
    assert api.post("/api/index/record", json={"app": "bench", "entries": [{**KEY, "output_id": oid}]}).status_code == 200
    r = api.post("/api/index/lookup", json={"app": "bench", "keys": [KEY]})
    assert r.status_code == 200, r.text
    assert r.json()["hits"][0]["output_id"] == back
    assert api.post("/api/index/forget", json={"app": "bench", "output_ids": [back]}).json() == {"forgotten": 1}
    assert api.post("/api/index/lookup", json={"app": "bench", "keys": [KEY]}).json()["hits"] == [None]
