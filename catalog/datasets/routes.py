"""The Dataset Catalog's routes under /api (README "Dataset Catalog", "API"), added to the catalog service's
app by `add_dataset_routes`. Errors use the catalog's shape, {"error": {"message", "type", "code", ...}}, with extra
fields where a caller needs them: `size` on a 409 from an upload, `latest` on a stale parent, `missing` on missing
audio and `problems` on an invalid version."""

from __future__ import annotations

import contextlib
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import anyio
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ..db import DB
from . import SCHEMA, DatasetError, gc
from .store import COPY_CHUNK, AudioStore
from .versions import Versions

if TYPE_CHECKING:
    from ..app import Catalog

T = TypeVar("T")


class MissingIn(BaseModel):
    sha256: list[str]


class HoldsIn(BaseModel):
    app: str = Field(min_length=1)
    add: list[str] = []
    remove: list[str] = []


class GcIn(BaseModel):
    dry_run: bool = True
    expect_files: int | None = Field(None, ge=0)  # the dry run's count, which a real run must not exceed


class UnpackIn(BaseModel):
    sha256: str


class DatasetIn(BaseModel):
    name: str = Field(min_length=1)
    language: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    app: str = Field(min_length=1)


class DatasetPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class UseIn(BaseModel):
    app: str = Field(min_length=1)
    ref: str = Field(min_length=1)


class AudioFileIn(BaseModel):
    path: str
    sha256: str


class AudioVersionIn(BaseModel):
    files: list[AudioFileIn] = Field(min_length=1)
    parent: int | None = None
    app: str = "dataset-catalog"
    force: bool = False


class DatasetCatalog:
    """The store, its database and the versions on it, one per catalog service."""

    def __init__(self, root: Path, events: Any):
        self.db = DB(Path(root) / "datasets.db", SCHEMA)
        self.store = AudioStore(Path(root), self.db)
        self.versions = Versions(self.db, self.store, events)


def _parent(v: str | None) -> int | None:
    if v is None or not v.strip() or v.strip().lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        raise DatasetError(422, f"parent must be a version number, not {v!r}", "invalid_request") from None


def _save(upload: UploadFile, store: AudioStore) -> Path:
    """An uploaded zip, copied under the store's tmp/ (the caller removes it)."""
    path = store.root / "tmp" / f"upload-{uuid.uuid4().hex}.zip"
    with open(path, "wb") as fh:
        shutil.copyfileobj(upload.file, fh, COPY_CHUNK)
    return path


@contextlib.asynccontextmanager
async def _in_thread(cm: contextlib.AbstractContextManager[T]) -> AsyncIterator[T]:
    """A blocking context manager entered and left on a worker thread, so its waits for the store's locks never
    stall the event loop."""
    value = await anyio.to_thread.run_sync(cm.__enter__)
    try:
        yield value
    except BaseException as exc:
        if not await anyio.to_thread.run_sync(cm.__exit__, type(exc), exc, exc.__traceback__):
            raise
    else:
        await anyio.to_thread.run_sync(cm.__exit__, None, None, None)


def _version_response(version: dict[str, Any], created: bool) -> JSONResponse:
    return JSONResponse(version, status_code=201 if created else 200)


def add_dataset_routes(app: FastAPI, cat: Catalog) -> DatasetCatalog:
    dc = DatasetCatalog(cat.settings.dataset_store, cat.events)
    app.state.datasets = dc
    store, versions = dc.store, dc.versions

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
        # Async to stream the body; the store's database calls run on a worker thread.
        async with _in_thread(store.upload(sha256, ext, offset)) as fh:
            if fh is not None:
                async for chunk in request.stream():
                    fh.write(chunk)
        return {"size": await anyio.to_thread.run_sync(store.upload_size, sha256)}

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

    @app.post("/api/store/gc")
    def store_gc(body: GcIn) -> dict[str, Any]:
        return gc.collect(store, body.dry_run, expect_files=body.expect_files)

    @app.post("/api/store/unpack")
    def store_unpack(body: UnpackIn) -> dict[str, Any]:
        return store.unpack(body.sha256)

    # ------------------------------------------------------------------------- datasets and versions (§7.2)

    @app.get("/api/datasets")
    def list_datasets(q: str | None = None, language: str | None = None, label_type: str | None = None,
                      app_: str | None = Query(None, alias="app")) -> list[dict[str, Any]]:
        return versions.list(q, language, label_type, app_)

    @app.post("/api/datasets", status_code=201)
    def create_dataset(body: DatasetIn) -> dict[str, Any]:
        return versions.create(body.name, body.app, body.language, body.description, body.tags)

    @app.post("/api/datasets/import")
    def import_package(file: UploadFile = File(...), dataset_id: str | None = Form(None),
                       app_: str = Form("dataset-catalog", alias="app"), preview: bool = Form(False)) -> JSONResponse:
        path = _save(file, store)
        try:
            out, created = versions.import_package(path, dataset_id or None, app_, file.filename or "", preview)
        finally:
            path.unlink(missing_ok=True)
        return JSONResponse(out) if preview else _version_response(out, created)

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        return versions.get(dataset_id)

    @app.patch("/api/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, body: DatasetPatch) -> dict[str, Any]:
        return versions.update(dataset_id, body.model_dump(exclude_unset=True))

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str) -> Response:
        versions.delete(dataset_id)
        return Response(status_code=204)

    @app.post("/api/datasets/{dataset_id}/versions")
    def publish(dataset_id: str, file: UploadFile = File(...), parent: str | None = Form(None),
                app_: str = Form(..., alias="app"), force: bool = Form(False)) -> JSONResponse:
        path = _save(file, store)
        try:
            return _version_response(*versions.publish(dataset_id, path, _parent(parent), app_, force))
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/datasets/{dataset_id}/versions/audio")
    def publish_audio(dataset_id: str, body: AudioVersionIn) -> JSONResponse:
        files = [f.model_dump() for f in body.files]
        return _version_response(*versions.publish_audio(dataset_id, files, body.parent, body.app, body.force))

    @app.get("/api/datasets/{dataset_id}/versions/{n}")
    def get_version(dataset_id: str, n: int) -> dict[str, Any]:
        return versions.version(dataset_id, n)

    @app.delete("/api/datasets/{dataset_id}/versions/{n}", status_code=204)
    def delete_version(dataset_id: str, n: int) -> Response:
        versions.delete_version(dataset_id, n)
        return Response(status_code=204)

    @app.get("/api/datasets/{dataset_id}/versions/{n}/files")
    def version_files(dataset_id: str, n: int, offset: int = Query(0, ge=0),
                      limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        return versions.files(dataset_id, n, offset, limit)

    @app.get("/api/datasets/{dataset_id}/versions/{n}/refs")
    def version_refs(dataset_id: str, n: int, path: str) -> dict[str, Any]:
        return versions.refs(dataset_id, n, path)

    @app.get("/api/datasets/{dataset_id}/versions/{n}/package")
    def version_package(dataset_id: str, n: int, audio: bool = False) -> StreamingResponse:
        name, chunks = versions.package(dataset_id, n, audio)
        return StreamingResponse(chunks, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/api/datasets/{dataset_id}/versions/{n}/uses", status_code=204)
    def add_use(dataset_id: str, n: int, body: UseIn) -> Response:
        versions.add_use(dataset_id, n, body.app, body.ref)
        return Response(status_code=204)

    @app.delete("/api/datasets/{dataset_id}/versions/{n}/uses", status_code=204)
    def remove_use(dataset_id: str, n: int, app_: str = Query(..., alias="app"), ref: str = Query(...)) -> Response:
        versions.remove_use(dataset_id, n, app_, ref)
        return Response(status_code=204)

    @app.get("/api/datasets/{dataset_id}/compare")
    def compare(dataset_id: str, a: int, b: int) -> dict[str, Any]:
        return versions.compare(dataset_id, a, b)

    return dc
