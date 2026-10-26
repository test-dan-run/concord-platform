"""The batch routes take an alias and act on the model's name, like every other route."""

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_an_override_set_through_an_alias_applies_to_the_model(api, cat):
    r = api.put("/api/models/default-tts/batch-override", json={"max_batch": 3})
    assert r.status_code == 200 and r.json()["max_batch"] == 3
    assert cat.profiler.start_env("kokoro-tts", None, 24.0) == {"AISVC_MAX_BATCH": "3"}
    by_name = api.get("/api/models/kokoro-tts/batch").json()
    by_alias = api.get("/api/models/default-tts/batch").json()
    assert by_alias["model"] == "kokoro-tts"
    assert by_name["override"]["max_batch"] == by_alias["override"]["max_batch"] == 3
    assert by_alias["limits"]["max_batch"] == 3 and by_alias["limits"]["source"]["max_batch"] == "override"
    assert api.delete("/api/models/default-tts/batch-override").status_code == 204
    assert cat.profiler.override("kokoro-tts") is None
    assert cat.db.all("SELECT model_id FROM batch_override") == []


def test_a_batch_profile_asked_for_through_an_alias_queues_the_name(api, cat):
    r = api.post("/api/models/default-tts/batch-profile")
    assert r.status_code == 202 and r.json() == {"model": "kokoro-tts", "measuring": "pending"}
    assert cat.profiler.pending == {"kokoro-tts": "pending"}


def test_the_batch_routes_still_refuse_what_they_refused(api):
    assert api.put("/api/models/default-chat/batch-override", json={"max_batch": 4}).status_code == 409  # vLLM
    assert api.delete("/api/models/default-vad/batch-override").status_code == 409  # CPU
    for r in (api.get("/api/models/no-such/batch"), api.put("/api/models/no-such/batch-override", json={"max_batch": 4}),
              api.delete("/api/models/no-such/batch-override"), api.post("/api/models/no-such/batch-profile")):
        assert r.status_code == 404
