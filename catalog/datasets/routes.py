"""The Dataset Catalog's routes under /api (DATASET_CATALOG_PLAN.md §7.1 and §7.2), added to the catalog service's
app by `add_dataset_routes`. Errors use the catalog's shape, {"error": {"message", "type", "code", ...}}, with extra
fields where a caller needs them, e.g. `size` on a 409 from an upload."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..db import DB
from . import SCHEMA, DatasetError
from .store import AudioStore

if TYPE_CHECKING:
    from ..app import Catalog


class MissingIn(BaseModel):
    sha256: list[str]


class HoldsIn(BaseModel):
    app: str = Field(min_length=1)
    add: list[str] = []
    remove: list[str] = []


class UnpackIn(BaseModel):
    sha256: str


class DatasetCatalog:
    """The store and its database, one per catalog service."""

    def __init__(self, root: Path, events: Any):
        self.db = DB(Path(root) / "datasets.db", SCHEMA)
        self.store = AudioStore(Path(root), self.db)


def add_dataset_routes(app: FastAPI, cat: Catalog) -> DatasetCatalog:
    dc = DatasetCatalog(cat.settings.dataset_store, cat.events)
    app.state.datasets = dc
    store = dc.store

    @app.exception_handler(DatasetError)
    async def _dataset_error(request: Request, exc: DatasetError) -> JSONResponse:
        return JSONResponse({"error": {"message": str(exc), "type": exc.type, "code": exc.status, **exc.extra}},
                            status_code=exc.status)

    # ------------------------------------------------------------------------- audio store (§7.1)

    @app.post("/api/store/missing")
    def store_missing(body: MissingIn) -> dict[str, Any]:
        return {"missing": store.missing(body.sha256)}

    @app.put("/api/store/audio/{sha256}")
    async def store_put(sha256: str, request: Request, ext: str = "", offset: int = Query(0, ge=0)) -> dict[str, Any]:
        with store.upload(sha256, ext, offset) as fh:
            if fh is not None:
                async for chunk in request.stream():
                    fh.write(chunk)
        return {"size": store.upload_size(sha256)}

    @app.post("/api/store/audio/{sha256}/commit")
    def store_commit(sha256: str) -> dict[str, Any]:
        return store.commit(sha256)

    @app.get("/api/store/audio/{sha256}")
    def store_get(sha256: str) -> FileResponse:
        rec = store.require(sha256)
        return FileResponse(store.path(sha256), media_type=store.media_type(rec))

    @app.post("/api/store/holds")
    def store_holds(body: HoldsIn) -> dict[str, Any]:
        return {"held": store.holds(body.app, body.add, body.remove)}

    @app.get("/api/store/stats")
    def store_stats() -> dict[str, Any]:
        return store.stats()

    @app.post("/api/store/unpack")
    def store_unpack(body: UnpackIn) -> dict[str, Any]:
        return store.unpack(body.sha256)

    return dc
