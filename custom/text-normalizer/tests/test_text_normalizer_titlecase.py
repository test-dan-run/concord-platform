"""text-normalizer: a sentence starts with the letter's titlecase, which differs from its uppercase for a few
letters: Georgian has no capitals at the start of a sentence, and ß and ᾳ have their own titlecase forms."""

from pathlib import Path

import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
norm = load_server(SERVICE_DIR, "normalize")


@pytest.mark.parametrize(("text", "expected"), [
    ("გამარჯობა. როგორ ხარ", "გამარჯობა. როგორ ხარ."),
    ("ᾳδω", "ᾼδω."),
    ("ßa", "Ssa."),
])
def test_sentence_starts_take_the_titlecase(text, expected):
    assert norm.normalize(text) == expected
