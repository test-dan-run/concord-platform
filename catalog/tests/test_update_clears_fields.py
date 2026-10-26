"""PATCH /api/models/{id} writes every field it is given, so null clears one; register leaves out what it is not given."""

import pytest
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


def test_null_clears_a_field_and_a_field_left_out_stays(api):
    r = api.post("/api/models/qwen3-8b/register",
                 json={"display_name": "Q8", "tags": ["a"], "parent_id": "fast-chat"})
    assert r.status_code == 201 and r.json()["parent_id"] == "qwen3-4b-instruct-2507"
    r = api.patch("/api/models/qwen3-8b", json={"tags": None})
    assert r.json()["tags"] == [] and r.json()["display_name"] == "Q8"
    assert r.json()["parent_id"] == "qwen3-4b-instruct-2507"
    # what the UI's model page sends when "Derived from" is emptied
    r = api.patch("/api/models/qwen3-8b", json={"display_name": "Q8", "tags": ["a"], "parent_id": None})
    assert r.status_code == 200 and r.json()["parent_id"] is None
    assert r.json()["display_name"] == "Q8" and r.json()["tags"] == ["a"]


def test_register_again_without_fields_keeps_them(api):
    api.post("/api/models/qwen3-8b/register", json={"display_name": "Q8", "tags": ["a"], "parent_id": "fast-chat"})
    r = api.post("/api/models/qwen3-8b/register", json={"parent_id": None})
    assert r.status_code == 201
    assert (r.json()["display_name"], r.json()["tags"], r.json()["parent_id"]) == ("Q8", ["a"], "qwen3-4b-instruct-2507")
    r = api.post("/api/models/qwen3-8b/register")
    assert (r.json()["display_name"], r.json()["tags"], r.json()["parent_id"]) == ("Q8", ["a"], "qwen3-4b-instruct-2507")


def test_a_hosted_model_keeps_its_fields_on_register_and_clears_them_on_patch(api):
    hosted = {"id": "hosted-asr", "contract": "openai", "api": "audio.transcriptions",
              "base_url": "https://api.example.com/v1", "model": "whisper-1", "auth_type": "bearer", "key": "sk-x",
              "display_name": "H", "tags": ["h"], "parent_id": "default-asr", "data_leaves_network": True}
    assert api.post("/api/hosted", json=hosted).status_code == 201
    r = api.post("/api/models/hosted-asr/register", json={})
    assert (r.json()["display_name"], r.json()["tags"], r.json()["parent_id"]) == ("H", ["h"], "whisper-large-v3")
    r = api.patch("/api/models/hosted-asr", json={"parent_id": None})
    assert r.json()["parent_id"] is None and r.json()["tags"] == ["h"]
