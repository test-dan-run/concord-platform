from __future__ import annotations

import shutil

import pytest
from factories import ServicesDir

from tools.schema import PLATFORM_DIR

SERVICES = PLATFORM_DIR / "services"


@pytest.fixture(autouse=True)
def _direct_stack(monkeypatch):
    """stack.py must drive (mocked) Docker in tests, never a model catalog running on this machine."""
    monkeypatch.setenv("AI_STACK_DIRECT", "1")


@pytest.fixture
def services(tmp_path) -> ServicesDir:
    d = tmp_path / "services"
    shutil.copytree(SERVICES, d)
    return ServicesDir(d)


@pytest.fixture
def empty_services(tmp_path) -> ServicesDir:
    d = tmp_path / "services"
    d.mkdir()
    return ServicesDir(d)
