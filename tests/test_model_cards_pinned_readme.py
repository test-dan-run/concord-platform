"""A README.md in a weights folder is the model's card only when SOURCE.txt there records where the weights came from.

The rule has one owner, `model_cards.pinned_readme`, and both the catalog's card and `just fetch-cards` follow it.
"""

import pytest

from catalog import cards
from tools import model_cards
from tools.schema import load_platform

NAME = "cohere-transcribe-03-2026"  # an engine model whose manifest comment also names a source


@pytest.fixture
def platform(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_ASSETS", str(tmp_path))
    p = load_platform()
    folder = model_cards.weights_folder(p, p.get(NAME))
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("---\nlicense: mit\n---\n# Acme ASR\nRead from the weights folder.\n")
    return p


def fetch_cards(monkeypatch, capsys) -> str:
    def no_fetch(p, src, http):
        raise RuntimeError("would fetch")

    monkeypatch.setattr(model_cards, "fetch", no_fetch)
    model_cards.main([NAME])
    return capsys.readouterr().out


def test_a_readme_without_source_txt_is_not_the_card(platform, monkeypatch, capsys):
    s = platform.get(NAME)
    folder = model_cards.weights_folder(platform, s)
    assert model_cards.source_of(platform, s).origin == "manifest comment"
    assert model_cards.pinned_readme(folder, model_cards.source_of(platform, s)) is None
    card = cards.build(platform, s)
    assert card["sections"] == [] and card["license"]["name"] is None
    assert any(f"fetch-cards {NAME}" in g for g in card["gaps"])
    assert "would fetch" in fetch_cards(monkeypatch, capsys)


def test_a_readme_with_source_txt_is_the_card(platform, monkeypatch, capsys):
    s = platform.get(NAME)
    folder = model_cards.weights_folder(platform, s)
    (folder / "SOURCE.txt").write_text("repo: acme/asr\nrevision: d552dad\n")
    src = model_cards.source_of(platform, s)
    assert model_cards.pinned_readme(folder, src) == folder / "README.md"
    up = cards.build(platform, s)["sections"][0]
    assert (up["key"], up["origin"], up["markdown"]) == ("upstream", "weights folder", "Read from the weights folder.")
    assert "on disk" in fetch_cards(monkeypatch, capsys)


def test_no_readme_or_no_source_is_no_pinned_readme(platform):
    s = platform.get(NAME)
    folder = model_cards.weights_folder(platform, s)
    (folder / "SOURCE.txt").write_text("repo: acme/asr\nrevision: d552dad\n")
    src = model_cards.source_of(platform, s)
    assert model_cards.pinned_readme(folder, None) is None and model_cards.pinned_readme(None, src) is None
    (folder / "README.md").unlink()
    assert model_cards.pinned_readme(folder, src) is None
