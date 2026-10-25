"""The catalog service: REST API under /api, the gateway under /m/ and /v2/, the UI at /."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tools.schema import CUSTOM_ENGINE

from . import cards
from .builds import Builds
from .datasets.routes import add_dataset_routes
from .db import DB
from .events import Events
from .exporter import Exporter
from .gateway import Gateway, error
from .hosted import HostedStore, Keys
from .index import IndexKeyError, OutputIndex
from .planner import PlanError, plan
from .registry import Registry
from .runtime import DockerRuntime, Runtime
from .batching import BatchProfiler, full_budget_gib
from .scheduler import Scheduler, SchedulerError
from .settings import Settings
from .verifier import Verifier

log = logging.getLogger("catalog")


@dataclass
class Catalog:
    settings: Settings
    db: DB
    events: Events
    runtime: Runtime
    builds: Builds
    scheduler: Scheduler
    verifier: Verifier
    registry: Registry
    index: OutputIndex
    gateway: Gateway
    hosted: HostedStore
    exporter: Exporter
    profiler: BatchProfiler

    def start(self) -> None:
        self.exporter.recover()
        self.scheduler.start()
        self.builds.prewarm(self.scheduler.platform())
        self.verifier.start()
        self.profiler.start()

    def shutdown(self) -> None:
        self.profiler.shutdown()
        self.verifier.shutdown()
        self.scheduler.shutdown()


def build_catalog(settings: Settings | None = None, runtime: Runtime | None = None) -> Catalog:
    settings = settings or Settings()
    db = DB(settings.db_path)
    events = Events()
    runtime = runtime or DockerRuntime()
    builds = Builds(db, runtime)
    hosted = HostedStore(db, Keys(settings.master_key_file))
    scheduler = Scheduler(db, settings, runtime, builds, events)
    scheduler.hosted_ids = hosted.ids
    verifier = Verifier(db, scheduler, auto=settings.verify_on_new_build, hosted=hosted,
                        hosted_reverify_s=settings.hosted_reverify_s)
    registry = Registry(db, scheduler, verifier, settings, hosted)
    profiler = BatchProfiler(db, scheduler, runtime)
    scheduler.batch_env = profiler.start_env
    return Catalog(settings, db, events, runtime, builds, scheduler, verifier, registry, OutputIndex(db),
                   Gateway(scheduler, settings, hosted), hosted,
                   Exporter(db, settings, scheduler, registry, runtime, events), profiler)


# ----------------------------------------------------------------------------- request bodies

class RegisterIn(BaseModel):
    display_name: str | None = None
    tags: list[str] | None = None
    parent_id: str | None = None


class HostedIn(BaseModel):
    """A hosted endpoint. `key` is write-only: no response ever carries it."""

    id: str | None = None
    display_name: str | None = None
    tags: list[str] | None = None
    parent_id: str | None = None
    contract: str | None = None
    api: str | None = None
    base_url: str | None = None
    model: str | None = None
    auth_type: str | None = None
    header_name: str | None = None
    key: str | None = Field(default=None, repr=False)
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    max_concurrency: int | None = Field(default=None, ge=1, le=256)
    max_upload_mb: float | None = Field(default=None, gt=0, le=10_000)
    data_leaves_network: bool = False


class ExportIn(BaseModel):
    models: list[str] = Field(min_length=1)
    destination: str = "bundle"
    registry: str | None = None
    images: bool = True
    weights: bool = True
    verification: bool = True


class LeaseIn(BaseModel):
    holder: str
    models: list[str] = Field(min_length=1)
    ttl_s: float | None = Field(default=None, gt=0, le=3600)


class BatchOverrideIn(BaseModel):
    max_batch: int | None = None  # most items in one batch (AISVC_MAX_BATCH)
    max_len_s: float | None = None  # most audio in one batch, items x the longest, in seconds (AISVC_MAX_BATCH_LEN)


class AppIn(BaseModel):
    name: str
    base_url: str


class LookupIn(BaseModel):
    app: str = ""
    keys: list[dict[str, Any]]


class RecordIn(BaseModel):
    app: str
    entries: list[dict[str, Any]]


class ForgetIn(BaseModel):
    app: str
    output_ids: list[Any]


def create_app(catalog: Catalog | None = None, *, start: bool = True) -> FastAPI:
    cat = catalog or build_catalog()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if start:
            cat.start()
        yield
        await cat.gateway.close()
        if start:
            cat.shutdown()

    app = FastAPI(title="Model Catalog", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.catalog = cat
    sched, reg = cat.scheduler, cat.registry

    @app.exception_handler(SchedulerError)
    async def _sched_error(request: Request, exc: SchedulerError) -> JSONResponse:
        return error(exc.status, str(exc), "catalog_error")

    @app.exception_handler(IndexKeyError)
    @app.exception_handler(PlanError)
    async def _bad_request(request: Request, exc: Exception) -> JSONResponse:
        return error(422, str(exc), "invalid_request")

    # ------------------------------------------------------------------------- platform and models

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        p = sched.platform()
        return {"ok": True, "services": len(p.services), "platform_error": sched.platform_error,
                "public_url": cat.settings.public_url}

    @app.get("/api/platform")
    def platform() -> dict[str, Any]:
        p = sched.platform()
        with sched.lock:
            # What each start was given (the whole GPU as measured), else the minimum to load.
            held = {m: sched.runs[m].given or sched.frac(p, m) for m in sorted(sched._occupying(p))}
            swaps = sched.swaps
        return {"gpu_total_gb": p.config.gpu_total_gb, "util_limit": p.config.util_limit,
                "reserve_gb": round((1 - p.config.util_limit) * p.config.gpu_total_gb, 2),
                "used_fraction": round(sum(held.values()), 4),
                "gpu_models": [{"id": m, "fraction": round(f, 4)} for m, f in held.items()],
                "swaps": swaps, "platform_error": sched.platform_error}

    @app.get("/api/models")
    def list_models(task: list[str] = Query(default=[]), tag: list[str] = Query(default=[]),
                    state: str | None = None, all: bool = False) -> list[dict[str, Any]]:  # noqa: A002
        return reg.records(task or None, tag or None, state, include_all=all)

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str) -> dict[str, Any]:
        return reg.record(model_id)

    @app.post("/api/models/{model_id}/register", status_code=201)
    def register(model_id: str, body: RegisterIn | None = None) -> dict[str, Any]:
        b = body or RegisterIn()
        return reg.register(model_id, b.display_name, b.tags, b.parent_id)

    @app.patch("/api/models/{model_id}")
    def update(model_id: str, body: RegisterIn) -> dict[str, Any]:
        return reg.update(model_id, body.model_dump(exclude_unset=True))

    @app.delete("/api/models/{model_id}/register", status_code=204)
    def unregister(model_id: str) -> Response:
        reg.unregister(model_id)
        return Response(status_code=204)

    @app.post("/api/models/{model_id}/verify", status_code=202)
    def verify(model_id: str) -> dict[str, Any]:
        rec = reg.verify(model_id)
        return rec["capabilities"] or {"status": "pending"}

    @app.get("/api/models/{model_id}/batch")
    def batch(model_id: str, vram_gb: float | None = Query(default=None, gt=0),
              item_s: float | None = Query(default=None, gt=0), item_tokens: int | None = Query(default=None, gt=0),
              n: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
        """The batch profile of the model's current build: the recommended batch for vram_gb (default: the full
        budget) and an item of item_s seconds (audio) or item_tokens tokens (vLLM), the memory batch `n` needs,
        and a table of lengths against 16, 24 and 32 GB and the full budget. For a custom GPU image, `limits` are
        the ones its next start gets at the full budget and `override` is what was set by hand. `running` is
        what the running service started with and reports now."""
        p = sched.platform()
        try:
            s = p.get(model_id)
        except KeyError:
            raise SchedulerError(404, f"unknown model {model_id!r}") from None
        with sched.lock:
            run = sched.runs.get(model_id)
            bk = run.build.build_key if run and run.build else None
        if bk is None:
            try:
                bk = cat.builds.current(p, p.get(model_id), None).build_key
            except Exception:  # noqa: BLE001
                bk = None
        full = full_budget_gib(p, sched)
        out = cat.profiler.view(model_id, bk, vram_gb=vram_gb, item_s=item_s, item_tokens=item_tokens, n=n,
                                full_gib=full)
        if s.gpu and s.engine == CUSTOM_ENGINE:
            out["limits"] = cat.profiler.limits(model_id, bk, full)
            out["override"] = cat.profiler.override(model_id)
        out["running"] = cat.profiler.in_use(p, s)
        return out

    def _custom_gpu(model_id: str) -> None:
        try:
            s = sched.platform().get(model_id)
        except KeyError:
            raise SchedulerError(404, f"unknown model {model_id!r}") from None
        if not s.gpu or s.engine != CUSTOM_ENGINE:
            raise SchedulerError(409, f"{model_id} is not a custom GPU image: the catalog does not set its batch "
                                      "limits" + (" (vLLM sizes its own batches)" if s.engine == "vllm" else ""))

    @app.put("/api/models/{model_id}/batch-override")
    def set_batch_override(model_id: str, body: BatchOverrideIn) -> dict[str, Any]:
        """Set the batch limits by hand for the model's next starts (a limit left out stays measured)."""
        _custom_gpu(model_id)
        try:
            return cat.profiler.set_override(model_id, body.max_batch, body.max_len_s)
        except ValueError as exc:
            raise SchedulerError(422, str(exc)) from None

    @app.delete("/api/models/{model_id}/batch-override", status_code=204)
    def clear_batch_override(model_id: str) -> Response:
        _custom_gpu(model_id)
        cat.profiler.clear_override(model_id)
        return Response(status_code=204)

    @app.post("/api/models/{model_id}/batch-profile", status_code=202)
    def batch_profile(model_id: str) -> dict[str, Any]:
        """Measure the model's batch profile now (it leases the model and waits its turn)."""
        try:
            sched.platform().get(model_id)
        except KeyError:
            raise SchedulerError(404, f"unknown model {model_id!r}") from None
        return {"model": model_id, "measuring": cat.profiler.enqueue(model_id)}

    @app.get("/api/models/{model_id}/card")
    def model_card(model_id: str) -> dict[str, Any]:
        h = cat.hosted.get(model_id) if model_id in cat.hosted else None
        if h is not None:
            return cards.build_hosted(h)
        p = sched.platform()
        return cards.build(p, sched.manifest(model_id))

    @app.post("/api/models/{model_id}/test")
    def test_model(model_id: str) -> dict[str, Any]:
        return reg.test_model(model_id)

    @app.post("/api/hosted/test")
    def test_hosted(body: HostedIn) -> dict[str, Any]:
        return reg.test_hosted(body.model_dump())

    @app.post("/api/hosted", status_code=201)
    def create_hosted(body: HostedIn) -> dict[str, Any]:
        return reg.create_hosted(body.model_dump())

    @app.patch("/api/hosted/{model_id}")
    def update_hosted(model_id: str, body: HostedIn) -> dict[str, Any]:
        return reg.update_hosted(model_id, body.model_dump(exclude_unset=True))

    @app.delete("/api/hosted/{model_id}", status_code=204)
    def delete_hosted(model_id: str) -> Response:
        reg.delete_hosted(model_id)
        return Response(status_code=204)

    @app.post("/api/models/{model_id}/start", status_code=202)
    def start_model(model_id: str, holder: str = "manual") -> dict[str, Any]:
        if model_id in cat.hosted:
            raise SchedulerError(409, f"{model_id} is a hosted endpoint: it is always up and cannot be started")
        name = sched.manifest(model_id).name
        r = sched.ensure_started(name, holder)
        return {"model": reg.record(name), "request": r.public() if r else None}

    @app.post("/api/models/{model_id}/stop", status_code=202)
    def stop_model(model_id: str, force: bool = False) -> dict[str, Any]:
        if model_id in cat.hosted:
            raise SchedulerError(409, f"{model_id} is a hosted endpoint: it has no container to stop")
        name = sched.manifest(model_id).name
        sched.stop_model(name, force)
        sched.wake()
        return reg.record(name)

    @app.get("/api/usage")
    def usage() -> list[dict[str, Any]]:
        rows = {r["model_id"]: r for r in cat.db.all("SELECT * FROM usage")}
        out = []
        for name in [s.name for s in sched.platform().services] + sorted(cat.hosted.ids()):
            v = sched.model_view(name)
            u = rows.get(name) or {}
            out.append({"id": name, "calls": u.get("calls", 0), "errors": u.get("errors", 0),
                        "in_flight": v["in_flight"], "last_used": reg.record(name)["last_used"]})
        return out

    # ------------------------------------------------------------------------- export

    @app.get("/api/export/models")
    def export_models() -> list[dict[str, Any]]:
        return cat.exporter.models()

    @app.post("/api/exports", status_code=202)
    def start_export(body: ExportIn) -> dict[str, Any]:
        return cat.exporter.start(body.model_dump())

    @app.get("/api/exports")
    def list_exports() -> list[dict[str, Any]]:
        return cat.exporter.jobs()

    @app.get("/api/exports/{job_id}")
    def get_export(job_id: str) -> dict[str, Any]:
        return cat.exporter.get(job_id)

    @app.delete("/api/exports/{job_id}", status_code=204)
    def delete_export(job_id: str) -> Response:
        cat.exporter.cancel_or_delete(job_id)
        return Response(status_code=204)

    @app.get("/api/exports/{job_id}/download")
    def download_export(job_id: str) -> StreamingResponse:
        name, chunks = cat.exporter.download(job_id)
        return StreamingResponse(chunks, media_type="application/x-tar",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ------------------------------------------------------------------------- leases

    @app.post("/api/leases", status_code=201)
    def take_lease(body: LeaseIn) -> dict[str, Any]:
        return sched.request(body.holder, body.models, "lease", body.ttl_s).public()

    @app.get("/api/leases")
    def list_leases() -> list[dict[str, Any]]:
        return [r.public() for r in sched.live_requests()]

    @app.get("/api/leases/{lease_id}")
    def get_lease(lease_id: str) -> dict[str, Any]:
        return sched.get(lease_id).public()

    @app.post("/api/leases/{lease_id}/heartbeat")
    def heartbeat(lease_id: str) -> dict[str, Any]:
        return sched.heartbeat(lease_id).public()

    @app.delete("/api/leases/{lease_id}", status_code=204)
    def release(lease_id: str) -> Response:
        sched.release(lease_id)
        sched.wake()
        return Response(status_code=204)

    # ------------------------------------------------------------------------- index, apps, plans

    @app.post("/api/index/lookup")
    def index_lookup(body: LookupIn) -> dict[str, Any]:
        return {"hits": cat.index.lookup(body.app, body.keys)}

    @app.post("/api/index/record")
    def index_record(body: RecordIn) -> dict[str, Any]:
        return {"recorded": cat.index.record(body.app, body.entries)}

    @app.post("/api/index/forget")
    def index_forget(body: ForgetIn) -> dict[str, Any]:
        return {"forgotten": cat.index.forget(body.app, body.output_ids)}

    @app.get("/api/index/stats")
    def index_stats() -> list[dict[str, Any]]:
        return cat.index.stats()

    @app.post("/api/apps")
    def register_app(body: AppIn) -> dict[str, Any]:
        return cat.index.register_app(body.name, body.base_url)

    @app.get("/api/apps")
    def list_apps() -> list[dict[str, Any]]:
        return [{"name": n, "base_url": u} for n, u in sorted(cat.index.apps().items())]

    @app.post("/api/jobs")
    def make_plan(body: dict[str, Any]) -> dict[str, Any]:
        return plan(sched, cat.index, body)

    # ------------------------------------------------------------------------- events

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q = cat.events.subscribe()

        async def stream():
            try:
                yield "retry: 3000\n\n"
                for rec in reg.records(include_all=True):  # a snapshot first, so clients need no separate fetch
                    yield _sse("model", {"id": rec["id"], "state": rec["state"], "state_detail": rec["state_detail"],
                                         "build_key": rec["build_key"], "leased_by": rec["leased_by"]})
                idle = 0.0
                while not await request.is_disconnected():
                    try:
                        event, data = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.25)
                        idle += 0.25
                        if idle >= 15:
                            idle = 0.0
                            yield ": keep-alive\n\n"
                        continue
                    idle = 0.0
                    yield _sse(event, data)
            finally:
                cat.events.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------------- gateway

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    @app.api_route("/m/{model_id}/{path:path}", methods=methods, include_in_schema=False)
    async def gateway(model_id: str, path: str, request: Request) -> Response:
        return await cat.gateway.forward(request, model_id, path)

    @app.api_route("/v2/models/{model_id}/{path:path}", methods=methods, include_in_schema=False)
    async def gateway_oip(model_id: str, path: str, request: Request) -> Response:
        return await cat.gateway.forward(request, model_id, f"v2/models/{model_id}/{path}")

    add_dataset_routes(app, cat)  # the Dataset Catalog (catalog/datasets): before the UI's catch-all route

    # ------------------------------------------------------------------------- UI

    ui = cat.settings.ui_dir
    if (ui / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=ui / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        if path.startswith(("api/", "m/", "v2/")):
            return error(404, f"no route {path!r}", "not_found")
        f = ui / path
        if path and f.is_file() and ui in f.resolve().parents:
            return FileResponse(f)
        index = ui / "index.html"
        if index.is_file():  # always revalidated: each UI build renames its scripts, so a cached page would load none
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return Response("The catalog UI is not built: run `just catalog-ui`. The API is at /api/docs.",
                        media_type="text/plain")

    return app


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    uvicorn.run(create_app(build_catalog(settings)), host=settings.bind, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
