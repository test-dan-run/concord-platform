"""Relative links in an upstream model card resolve against the card's folder, like a browser would."""

import pytest

from catalog import cards
from tools.model_cards import Source

GH = Source("github", "o/r", "v1", "sub", "weights.yaml")


@pytest.mark.parametrize("target,folder,url", [
    (".github/x.png", "", "https://github.com/o/r/blob/v1/.github/x.png"),  # a hidden folder stays
    ("../docs/y.md", "a/b", "https://github.com/o/r/blob/v1/a/docs/y.md"),
    ("docs/./z.md#usage", "sub", "https://github.com/o/r/blob/v1/sub/docs/z.md#usage"),
    ("..x/y.md", "", "https://github.com/o/r/blob/v1/..x/y.md"),
])
def test_relative_links_resolve_against_the_folder(target, folder, url):
    assert cards.rewrite_links(f"[a]({target})", GH, "v1", folder) == f"[a]({url})"


def test_a_link_out_of_the_repo_is_left_alone():
    assert cards.rewrite_links("[a](../docs/y.md)", GH, "v1", "") == "[a](../docs/y.md)"
