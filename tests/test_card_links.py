"""Relative links in an upstream model card resolve against the card's folder, like a browser would."""

import pytest

from catalog import cards
from tools.model_cards import Source

GH = Source("github", "o/r", "v1", "sub", "weights.yaml")
HF = Source("huggingface", "acme/asr", "abc", "", "SOURCE.txt")


@pytest.mark.parametrize("target,folder,url", [
    (".github/x.png", "", "https://github.com/o/r/blob/v1/.github/x.png"),
    ("../docs/y.md", "a/b", "https://github.com/o/r/blob/v1/a/docs/y.md"),
    ("./b.png", "sub", "https://github.com/o/r/blob/v1/sub/b.png"),
    ("docs/./z.md#usage", "sub", "https://github.com/o/r/blob/v1/sub/docs/z.md#usage"),
    ("..x/y.md", "", "https://github.com/o/r/blob/v1/..x/y.md"),
])
def test_relative_links_resolve_against_the_folder(target, folder, url):
    assert cards.rewrite_links(f"[a]({target})", GH, "v1", folder) == f"[a]({url})"


@pytest.mark.parametrize("target,folder", [("../docs/y.md", ""), ("../../y.md", "sub")])
def test_a_link_out_of_the_repo_is_left_alone(target, folder):
    assert cards.rewrite_links(f"[a]({target})", GH, "v1", folder) == f"[a]({target})"


def test_hugging_face_images_keep_their_hidden_folder():
    assert cards.rewrite_links("![x](.assets/fig.png)", HF, "abc") == "![x](https://huggingface.co/acme/asr/resolve/abc/.assets/fig.png)"
