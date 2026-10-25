"""text-normalizer: rule 3 capitalises a sentence that starts with any letter, not only a to z."""

from pathlib import Path

import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
norm = load_server(SERVICE_DIR, "normalize")


@pytest.mark.parametrize(("text", "expected"), [
    ("élan vital. über alles. ωmega", "Élan vital. Über alles. Ωmega."),
    ("ångström! ça va? être", "Ångström! Ça va? Être."),
    ("привет мир. как дела", "Привет мир. Как дела."),
    ("3 apples. _x and 4 pears", "3 apples. _x and 4 pears."),  # digits and _ are not letters
    ("日本語の文. こんにちは", "日本語の文. こんにちは."),        # scripts without case stay as they are
])
def test_non_ascii_sentence_starts(text, expected):
    assert norm.normalize(text) == expected


def test_capitalize_false_leaves_them():
    assert norm.normalize("élan. über", norm.Options(capitalize=False)) == "élan. über."


def test_over_http():
    with running(app_mod.build_app()) as c:
        r = c.post("/v1/text/transform", json={"model": "text-normalizer", "input": ["élan vital. über alles"]})
    assert r.json()["data"][0]["output"] == "Élan vital. Über alles."
