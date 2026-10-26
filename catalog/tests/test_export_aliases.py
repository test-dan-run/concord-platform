"""An export of a model the platform does not know is a 404. (A name and its alias export the model once: see
test_export.py.)"""

from export_helpers import api, images  # noqa: F401  (fixtures)


def test_an_unknown_model_is_a_404(api, images):
    r = api.post("/api/exports", json={"models": ["silero-vad", "no-such-model"]})
    assert r.status_code == 404 and "no-such-model" in r.json()["error"]["message"]
