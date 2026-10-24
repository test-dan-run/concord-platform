"""Pieces shared by every app factory: readiness state, background loading, /health, CORS."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger("aisvc")

LoadFn = Callable[[], Any]


class ServiceState:
    """Readiness of one service. `load` runs in a background thread so /health can say 503 meanwhile."""

    def __init__(self, name: str):
        self.name = name
        self.ready = False
        self.error: str | None = None
        self.started = time.time()
        self._loaded = threading.Event()

    def run_load(self, load: LoadFn | None) -> None:
        try:
            if load is not None:
                log.info("loading %s ...", self.name)
                load()
            self.ready = True
            log.info("%s ready after %.1fs", self.name, time.time() - self.started)
        except Exception as exc:  # surfaced through /health; the container stays unhealthy
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("loading %s failed", self.name)
        finally:
            self._loaded.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until loading finished (ready or failed). Handy in tests."""
        self._loaded.wait(timeout)
        return self.ready


def create_base_app(name: str, *, title: str, load: LoadFn | None, background: bool = True) -> FastAPI:
    """FastAPI app with permissive CORS, GET /health and model loading on startup.

    No auth: every service is personal-use and bound to 127.0.0.1 by the platform.
    """
    state = ServiceState(name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if background and load is not None:
            threading.Thread(target=state.run_load, args=(load,), name=f"load-{name}", daemon=True).start()
        else:
            state.run_load(load)
        yield

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.svc = state
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health", tags=["platform"])
    def health() -> JSONResponse:
        if state.ready:
            return JSONResponse({"status": "ok", "name": name})
        body = {"status": "error" if state.error else "loading", "name": name}
        if state.error:
            body["error"] = state.error
        return JSONResponse(body, status_code=503)

    return app
