"""Fixtures for export tests: two models whose images share a layer, and the API. Import the fixtures by name."""

from __future__ import annotations

import pytest
from catalog_helpers import wait_builds
from fastapi.testclient import TestClient

from catalog.app import create_app

PAIR = ["silero-vad", "pyannote-diarization"]


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False)) as c:
        yield c


@pytest.fixture
def images(cat, runtime):
    """The two models' images share a 5000-byte base layer; pyannote's weights.yaml files are fetched."""
    import yaml

    from tools import identity

    p = cat.scheduler.platform()
    s = p.get("pyannote-diarization")
    for f in yaml.safe_load(identity.weights_source(p, s).path.read_text())["files"]:
        target = p.weights_dir(s) / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w" * 100)
    a, b = (p.image(p.get(m)) for m in PAIR)
    runtime.layers[a] = [("base", 5000), ("vad", 1000)]
    runtime.layers[b] = [("base", 5000), ("diar", 2000)]
    wait_builds(cat, PAIR)
    return a, b


def finish(cat, api, job_id, timeout=10.0):
    cat.exporter.wait(timeout)
    return api.get(f"/api/exports/{job_id}").json()
