"""POST /api/jobs takes aliases like every other route, and rejects a malformed body with 422."""

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


def test_a_staged_plan_takes_an_alias_and_gives_the_name(api):
    r = api.post("/api/jobs", json={"stages": [{"name": "a", "models": ["default-vad"]}]})
    assert r.status_code == 200, r.text
    assert r.json()["steps"][0]["model"] == "silero-vad"


def test_a_co_resident_plan_takes_aliases_and_gives_names(api):
    body = {"pipeline": "co-resident",
            "stages": [{"name": "seg", "models": ["default-vad"]},
                       {"name": "asr", "models": ["default-asr"], "after": "seg"}]}
    r = api.post("/api/jobs", json=body)
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert {s["model"] for s in steps} == {"silero-vad", "whisper-large-v3"}
    assert {s["alongside"] for s in steps if "alongside" in s} <= {"silero-vad", "whisper-large-v3"}


@pytest.mark.parametrize("body", [
    {"stages": "abc"},
    {"stages": [["x"]]},
    {"stages": ["x"]},
    {"stages": {"name": "a"}},
    {"stages": [{"name": "a", "models": "silero-vad"}]},
    {"stages": [{"name": "a", "models": [1]}]},
    {"stages": [{"name": "a", "models": [["silero-vad"]]}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": "nope"}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": {"silero-vad": "nope"}}]},
    {"stages": [{"name": "a", "models": ["silero-vad"], "keys": {"silero-vad": ["nope"]}}]},
])
def test_a_malformed_plan_body_is_rejected(api, body):
    r = api.post("/api/jobs", json=body)
    assert r.status_code == 422, (body, r.status_code, r.text)
