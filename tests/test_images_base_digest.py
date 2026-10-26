"""images.base_digest: a base is stale only when a file its own build reads changes."""

import shutil

import pytest

from tools import images
from tools.schema import load_bases, load_platform


@pytest.fixture
def bases(tmp_path):
    d = tmp_path / "_bases"
    shutil.copytree(load_platform().root / "custom" / "_bases", d)
    return d


def digests(d):
    return {name: images.base_digest(d, spec) for name, spec in load_bases(load_platform().root).items()}


def test_editing_one_dockerfile_leaves_the_other_bases_current(bases):
    before = digests(bases)
    (bases / "Dockerfile.py-cuda").write_text((bases / "Dockerfile.py-cuda").read_text() + "\n# note\n")
    after = digests(bases)
    assert after["py-cpu"] == before["py-cpu"]
    assert after["py-cuda"] != before["py-cuda"] and after["py-cuda-torch211"] != before["py-cuda-torch211"]


@pytest.mark.parametrize("path", ["common/README.md", "bases.yaml", "common/aisvc/__pycache__/batch.cpython-312.pyc",
                                  "common/stray.pyc"])
def test_files_the_build_leaves_out_do_not_change_the_digest(bases, path):
    before = digests(bases)
    f = bases / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes((f.read_bytes() if f.exists() else b"") + b"\n# edited\n")
    assert digests(bases) == before


@pytest.mark.parametrize("path", ["requirements-base.txt", "common/aisvc/batch.py", "common/pyproject.toml",
                                  ".dockerignore", "common/aisvc/new_module.py"])
def test_files_the_build_copies_change_every_digest(bases, path):
    before = digests(bases)
    f = bases / path
    f.write_bytes((f.read_bytes() if f.exists() else b"") + b"\n# edited\n")
    after = digests(bases)
    assert all(after[n] != before[n] for n in before)
