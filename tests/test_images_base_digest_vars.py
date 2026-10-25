"""base_digest refuses a COPY source with a build variable or a heredoc, and a Dockerfile that changes its escape
character: it would read them differently from Docker and miss the files they copy."""

import pytest

from tools import images


def base(tmp_path, dockerfile: str):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "x.py").write_text("x = 1\n")
    (tmp_path / "Dockerfile.py-cpu").write_text(dockerfile)
    return tmp_path


@pytest.mark.parametrize("copy", ["COPY ${SRC}/ /tmp/aisvc/", "COPY $SRC/ /tmp/aisvc/", "COPY <<EOF /tmp/x\nx\nEOF"])
def test_a_copy_source_it_cannot_resolve_is_refused(tmp_path, copy):
    root = base(tmp_path, f"FROM python:3.12-slim\nARG SRC=common\n{copy}\n")
    with pytest.raises(ValueError, match="one-line COPY/ADD"):
        images.base_digest(root, {"dockerfile": "Dockerfile.py-cpu"})


def test_an_escape_directive_is_refused(tmp_path):
    root = base(tmp_path, "# escape=`\nFROM python:3.12-slim\nCOPY common/ `\n    /tmp/aisvc/\n")
    with pytest.raises(ValueError, match="escape"):
        images.base_digest(root, {"dockerfile": "Dockerfile.py-cpu"})


def test_an_arg_used_elsewhere_is_fine(tmp_path):
    root = base(tmp_path, "ARG BASE=python:3.12-slim\nFROM ${BASE}\nCOPY common/ /tmp/aisvc/\n")
    before = images.base_digest(root, {"dockerfile": "Dockerfile.py-cpu"})
    (root / "common" / "x.py").write_text("x = 2\n")
    assert images.base_digest(root, {"dockerfile": "Dockerfile.py-cpu"}) != before
