"""base_digest reads the files a base's COPY lines name; a COPY it cannot read must fail, not be skipped."""

import pytest

from tools import images


@pytest.mark.parametrize("copy", ["COPY common/ \\\n    /tmp/aisvc/", 'COPY ["common/", "/tmp/aisvc/"]'])
def test_a_copy_form_it_cannot_read_is_refused(tmp_path, copy):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "x.py").write_text("x = 1\n")
    (tmp_path / "Dockerfile.py-cpu").write_text(f"FROM python:3.12-slim\n{copy}\n")
    with pytest.raises(ValueError, match="one-line COPY/ADD"):
        images.base_digest(tmp_path, {"dockerfile": "Dockerfile.py-cpu"})


def test_a_one_line_copy_is_hashed(tmp_path):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "x.py").write_text("x = 1\n")
    (tmp_path / "Dockerfile.py-cpu").write_text("FROM python:3.12-slim\nCOPY --chown=1000:1000 common/ /tmp/aisvc/\n")
    before = images.base_digest(tmp_path, {"dockerfile": "Dockerfile.py-cpu"})
    (tmp_path / "common" / "x.py").write_text("x = 2\n")
    assert images.base_digest(tmp_path, {"dockerfile": "Dockerfile.py-cpu"}) != before
