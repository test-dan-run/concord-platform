"""Run conformance checks in-process against a FastAPI app (no Docker, no network)."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any


def service_meta(name: str) -> dict[str, Any]:
    """Endpoint-map entry for a service, built straight from its manifest (no render needed)."""
    from tools.render import endpoint_entry
    from tools.schema import load_platform

    p = load_platform()
    return endpoint_entry(p, p.get(name))


def load_server(service_dir: Path, module: str = "app") -> ModuleType:
    """Import <service_dir>/server/<module>.py under a unique package name.

    Every custom service has a package called `server`; importing them under distinct names lets
    all their tests run in one pytest session.
    """
    service_dir = Path(service_dir)
    pkg = "svc_" + service_dir.name.replace("-", "_") + "_server"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(pkg, service_dir / "server" / "__init__.py",
                                                      submodule_search_locations=[str(service_dir / "server")])
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{pkg}.{module}")


@contextmanager
def running(app: Any, timeout: float = 30.0) -> Iterator[Any]:
    """TestClient with the app's lifespan started and background loading finished."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        state = getattr(app.state, "svc", None)
        if state is not None and not state.wait(timeout):
            raise RuntimeError(f"service failed to become ready: {state.error}")
        yield client
