"""base_digest never guesses: a COPY/ADD, RUN --mount or .dockerignore pattern it cannot read the way Docker
does is refused, and the forms it reads match Docker's (any case, cleaned patterns, "! pattern")."""

import pytest

from tools import images


def context(tmp_path, instruction, ignore=""):
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "pyproject.toml").write_text("a\n")
    (tmp_path / "requirements-base.txt").write_text("numpy\n")
    (tmp_path / "Dockerfile.py-cpu").write_text(f"FROM python:3.12-slim\n{instruction}\n")
    (tmp_path / ".dockerignore").write_text(ignore)
    return lambda: images.base_digest(tmp_path, {"dockerfile": "Dockerfile.py-cpu"})


@pytest.mark.parametrize("copy", [
    "COPY common/*.toml /tmp/aisvc/",
    "COPY common/[p]yproject.toml /tmp/aisvc/",
    "COPY /requirements-base.txt /tmp/",
    "COPY ../requirements-base.txt /tmp/",
    "COPY --from=builder /opt/venv /opt/venv",
    "COPY requirements-base.txt /tmp/\\\n    common/ /tmp/aisvc/",
    "COPY requirements-base.txt /tmp/ \\  \n    common/ /tmp/aisvc/",
    'COPY ["common/", "/tmp/aisvc/"]',
], ids=["star", "class", "absolute", "parent", "from", "attached-backslash", "backslash-then-spaces", "json"])
def test_a_copy_it_cannot_read_is_refused(tmp_path, copy):
    with pytest.raises(ValueError, match="one-line COPY/ADD"):
        context(tmp_path, copy)()


@pytest.mark.parametrize("copy", ["Add requirements-base.txt /tmp/", "COPY ./requirements-base.txt /tmp/"])
def test_a_copy_in_any_case_is_hashed(tmp_path, copy):
    digest = context(tmp_path, copy)
    before = digest()
    (tmp_path / "requirements-base.txt").write_text("torch\n")
    assert digest() != before


@pytest.mark.parametrize("run", [
    "RUN --mount=type=bind,source=common,target=/src pip install /src",
    "RUN --mount=source=common,target=/src pip install /src",
    "RUN \\\n    --mount=type=bind,source=common,target=/src pip install /src",
], ids=["bind", "default-type", "continued"])
def test_a_run_that_binds_context_files_is_refused(tmp_path, run):
    with pytest.raises(ValueError, match="RUN --mount"):
        context(tmp_path, run)()


def test_a_run_with_a_cache_mount_is_fine(tmp_path):
    context(tmp_path, "RUN --mount=type=cache,target=/root/.cache pip install numpy \\\n    && echo done")()


def rules(tmp_path, text):
    (tmp_path / ".dockerignore").write_text(text)
    return images._ignore_rules(tmp_path)


def test_ignore_patterns_are_cleaned_like_docker_does(tmp_path):
    r = rules(tmp_path, "./common/README.md\n/common//notes/\ncommon/x/../old.txt\n")
    assert images._ignored("common/README.md", r)
    assert images._ignored("common/notes/a.md", r)
    assert images._ignored("common/old.txt", r)
    assert not images._ignored("common/pyproject.toml", r)


def test_spaces_after_the_bang_are_dropped(tmp_path):
    r = rules(tmp_path, "common/*.md\n!  common/KEEP.md\n")
    assert not images._ignored("common/KEEP.md", r)
    assert images._ignored("common/README.md", r)


def test_double_star_as_a_whole_name_still_works(tmp_path):
    r = rules(tmp_path, "**/__pycache__\ncommon/**/*.pyc\nbuild/**\n")
    assert images._ignored("common/aisvc/__pycache__/x.pyc", r) and images._ignored("__pycache__", r)
    assert images._ignored("common/a/b/c.pyc", r) and images._ignored("common/c.pyc", r)
    assert images._ignored("build/x/y", r) and not images._ignored("common/x.py", r)


@pytest.mark.parametrize("pattern", ["common/[Rr]EADME.md", "common/a**b", "common/a\\*b"])
def test_an_ignore_pattern_it_cannot_translate_is_refused(tmp_path, pattern):
    digest = context(tmp_path, "COPY common/ /tmp/aisvc/", ignore=f"{pattern}\n")
    with pytest.raises(ValueError, match=".dockerignore"):
        digest()
