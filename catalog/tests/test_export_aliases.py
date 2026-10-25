"""An export names each model once, whether it was asked for by name, by alias or both."""

import json

from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)


def test_a_name_and_its_alias_export_the_model_once(api, cat, images, settings):
    r = api.post("/api/exports", json={"models": ["silero-vad", "default-vad", "pyannote-diarization"]})
    assert r.status_code == 202 and r.json()["request"]["models"] == PAIR
    job = finish(cat, api, r.json()["id"])
    assert job["state"] == "done", job
    manifest = json.loads((settings.export_dir / job["name"] / "manifest.json").read_text())
    assert [m["id"] for m in manifest["models"]] == PAIR


def test_an_unknown_model_is_a_404(api, images):
    r = api.post("/api/exports", json={"models": ["silero-vad", "no-such-model"]})
    assert r.status_code == 404 and "no-such-model" in r.json()["error"]["message"]
