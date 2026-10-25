"""Gateway: inference calls by model id, forwarded unchanged to the model service.

`/m/{id}/<path>` -> `<service root>/<path>`; `/v2/models/{id}[/<path>]` -> the same service's OIP route.
A stopped model is started on demand and the call is held until it is healthy (2 x its measured cold
start, or gateway_unmeasured_wait_s), then 503 `model_starting`. Calls in flight keep a model from
being stopped.

A hosted endpoint is always up: its branch skips the scheduler's start logic, refuses bodies over the
upload limit (413), waits for one of its concurrency slots (503 after its timeout), swaps whatever auth
the caller sent for the stored key, and forwards the body unchanged to the provider URL. A successful
reply streams back unchanged; an error reply is read whole and the key redacted from it.
"""

from __future__ import annotations

import asyncio
import logging
import time

import anyio
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import hosted as hosted_mod
from .hosted import Hosted, HostedStore
from .scheduler import Scheduler, SchedulerError
from .settings import Settings

log = logging.getLogger("catalog.gateway")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
              "transfer-encoding", "upgrade", "host"}
ERROR_BODY_LIMIT = 1024 * 1024  # a hosted provider's error reply is read whole up to this size, to redact the key


def error(status: int, message: str, type_: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": type_, "code": status}}, status_code=status,
                        headers=headers)


def _raw(resp: httpx.Response, drop: set[str], key: str | None = None) -> list[tuple[bytes, bytes]]:
    """The upstream's headers for a response's raw_headers, as it sent them: a header sent twice (two Set-Cookie)
    stays two, not one joined by ", ", and a value keeps its bytes (httpx decodes UTF-8 ones as UTF-8). Leaves out
    the names in `drop` and, given a key, a header that quotes it."""
    out = []
    for k, v in resp.headers.raw:
        text = v.decode("latin-1")  # for the key check only: one character per byte, so it never fails
        if k.decode("latin-1").lower() not in drop and hosted_mod.redact(text, key) == text:
            out.append((k.lower(), v))
    return out


class TooLarge(Exception):
    pass


class Gateway:
    def __init__(self, scheduler: Scheduler, settings: Settings, hosted: HostedStore | None = None):
        self.scheduler = scheduler
        self.settings = settings
        self.hosted = hosted
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))
        self._slots: dict[str, tuple[int, asyncio.Semaphore]] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def _wait_up(self, name: str) -> JSONResponse | None:
        """None once the model is up; an error response if it failed or took too long."""
        sched = self.scheduler
        if sched.is_up(name):
            return None
        p = sched.platform()
        cold, measured = sched.cold_start(p, name)
        limit = 2 * cold if measured else self.settings.gateway_unmeasured_wait_s
        req = await anyio.to_thread.run_sync(sched.ensure_started, name)
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if sched.is_up(name):
                return None
            if req is not None:
                try:
                    cur = sched.get(req.id)
                    if req.kind == "start" and req.holder == "gateway" and cur.state in ("queued", "starting"):
                        sched.heartbeat(req.id)
                    if cur.state == "failed":
                        return error(503, f"{name} failed to start: {cur.error}", "model_failed")
                    if cur.state in ("done", "released", "expired") and not sched.is_up(name):
                        req = await anyio.to_thread.run_sync(sched.ensure_started, name)
                except SchedulerError:
                    req = await anyio.to_thread.run_sync(sched.ensure_started, name)
            await asyncio.sleep(0.5)
        view = sched.model_view(name)
        detail = f" ({view['state_detail']})" if view["state_detail"] else ""
        return error(503, f"{name} is starting ({view['state']}{detail}); retry shortly", "model_starting",
                     {"Retry-After": "30"})

    def _slot(self, h: Hosted) -> asyncio.Semaphore:
        limit, sem = self._slots.get(h.id, (0, None))
        if sem is None or limit != h.max_concurrency:  # a changed limit applies at once
            sem = asyncio.Semaphore(h.max_concurrency)
            self._slots[h.id] = (h.max_concurrency, sem)
        return sem

    async def forward_hosted(self, request: Request, h: Hosted, path: str) -> Response:
        sched = self.scheduler
        limit = int(h.max_upload_mb * 1024 * 1024)
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > limit:
            return error(413, f"request body is {int(length) / 1e6:.1f} MB; {h.id} accepts at most {h.max_upload_mb:g} MB",
                         "payload_too_large")
        try:
            key = self.hosted.key(h) if self.hosted else None
        except SchedulerError as exc:
            return error(503, str(exc), "key_unavailable")
        sem = self._slot(h)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=h.timeout_s)
        except TimeoutError:
            return error(503, f"{h.id} is at its limit of {h.max_concurrency} concurrent call(s); retry shortly",
                         "concurrency_limit", {"Retry-After": "5"})
        sched.begin_call(h.id)
        ended = False

        def end(err: bool) -> None:
            nonlocal ended
            if not ended:
                ended = True
                sem.release()
                sched.end_call(h.id, err)

        async def body():
            seen = 0
            async for chunk in request.stream():
                seen += len(chunk)
                if seen > limit:
                    raise TooLarge
                yield chunk

        try:
            url = hosted_mod.provider_url(h, path)
            headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
            headers = hosted_mod.strip_auth(headers, h) + list(hosted_mod.auth_headers(h, key).items())
            upstream = self.client.build_request(request.method, url, params=request.query_params, headers=headers,
                                                 content=body(), timeout=httpx.Timeout(h.timeout_s, connect=10.0))
            try:
                resp = await self.client.send(upstream, stream=True)
            except TooLarge:
                end(True)
                return error(413, f"request body is over {h.id}'s limit of {h.max_upload_mb:g} MB", "payload_too_large")
            except httpx.HTTPError as exc:
                end(True)
                return error(502, f"{h.host} did not answer: {type(exc).__name__}: {hosted_mod.redact(str(exc), key)}",
                             "upstream_error")
            if resp.status_code >= 400:  # an error reply may quote the key: read it and redact it
                return await self._redacted(resp, h, key, end)
            return self._stream(resp, end)
        except BaseException:
            end(True)
            raise

    async def _redacted(self, resp: httpx.Response, h: Hosted, key: str | None, end) -> Response:
        """A provider's error reply, read whole (decompressed) with the key redacted from its body. A header
        that quotes the key is dropped (a header cannot carry the •••• mask)."""
        data = bytearray()
        try:
            async for chunk in resp.aiter_bytes():
                data += chunk
                if len(data) > ERROR_BODY_LIMIT:
                    break
        except httpx.HTTPError as exc:
            end(True)
            return error(502, f"{h.host} did not answer: {type(exc).__name__}: {hosted_mod.redact(str(exc), key)}",
                         "upstream_error")
        finally:
            await resp.aclose()
            end(resp.status_code >= 500)
        if len(data) > ERROR_BODY_LIMIT:
            return error(resp.status_code, f"{h.host} answered {resp.status_code} with an error body over 1 MiB; "
                                           "it is not forwarded", "upstream_error")
        body = hosted_mod.redact(data.decode("utf-8", "surrogateescape"), key).encode("utf-8", "surrogateescape")
        drop = HOP_BY_HOP | {"content-length", "content-encoding"}  # the body changed: Response sets its length
        out = Response(body, status_code=resp.status_code)
        out.raw_headers += _raw(resp, drop, key)
        return out

    def _stream(self, resp: httpx.Response, end) -> StreamingResponse:
        async def close() -> None:
            await resp.aclose()
            end(resp.status_code >= 500)

        async def body():
            try:
                if resp.is_stream_consumed:  # an in-memory transport already read it
                    yield resp.content
                else:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            finally:  # also runs when the client disconnects mid-stream, so in-flight counts never leak
                await close()

        out = StreamingResponse(body(), status_code=resp.status_code, background=BackgroundTask(close))
        out.raw_headers = _raw(resp, HOP_BY_HOP)
        return out

    async def forward(self, request: Request, model: str, path: str) -> Response:
        sched = self.scheduler
        if self.hosted is not None and model in self.hosted:  # hosted first: it never touches the start logic
            h = self.hosted.get(model)
            if h is not None:
                return await self.forward_hosted(request, h, path)
        try:
            name = sched.manifest(model).name
        except SchedulerError as exc:
            return error(404, str(exc), "model_not_found")
        sched.begin_call(name)
        ended = False

        def end(err: bool) -> None:
            nonlocal ended
            if not ended:
                ended = True
                sched.end_call(name, err)

        try:
            waited = await self._wait_up(name)
            if waited is not None:
                end(True)
                return waited
            url = f"{sched.upstream_url(name)}/{path.lstrip('/')}"
            headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
            upstream = self.client.build_request(request.method, url, params=request.query_params, headers=headers,
                                                 content=request.stream())
            try:
                resp = await self.client.send(upstream, stream=True)
            except httpx.HTTPError as exc:
                end(True)
                return error(502, f"{name} did not answer: {type(exc).__name__}: {exc}", "upstream_error")
            return self._stream(resp, end)
        except BaseException:
            end(True)
            raise
