"""Each custom image's Dockerfile defaults ARG BASE_IMAGE to the base its manifest builds on. `just build` passes the
base with --build-arg; a plain `docker build custom/<name>` uses the default, so a stale one builds on another base."""

import re

from tools.schema import load_platform


def test_every_dockerfile_defaults_to_its_manifests_base():
    p = load_platform()
    wrong = {}
    for s in p.services:
        if not s.is_custom:
            continue
        m = re.search(r"^ARG BASE_IMAGE=(\S+)", (p.root / s.build.context / "Dockerfile").read_text(), re.M)
        if m is None or m.group(1) != p.base_image(s):
            wrong[s.name] = (m.group(1) if m else None, p.base_image(s))
    assert wrong == {}
