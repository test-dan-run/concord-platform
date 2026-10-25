"""Client for the Model Catalog's API (platform/catalog/README.md): models, leases, output index, plans.

    from aicore.catalog import Catalog
    cat = Catalog()                                   # AICORE_CATALOG, else http://127.0.0.1:8700
    with cat.lease("bench:job-12", ["whisper-large-v3"], on_wait=print):
        ...                                           # aicore.client(...) calls go through the gateway

`FakeCatalog` has the same methods, in memory, for app tests.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from .endpoints import AicoreError, catalog_url

DEFAULT_URL = "http://127.0.0.1:8700"
HEARTBEAT_S = 30
TTL_S = 90
WHOLE = -1.0  # span_start = span_end = -1: the whole file
CHUNK = 500  # index keys per request

OnWait = Callable[[dict[str, Any]], None]


class CatalogError(AicoreError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}" if status else message)
        self.status, self.message = status, message


class LeaseFailed(CatalogError):
    """The lease failed, expired or was released before it became active."""


def params_hash(params: dict[str, Any]) -> str:
    """The 16-hex-character hash apps use for request parameters in the output index."""
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def span(start: float | None, end: float | None) -> tuple[float, float]:
    """Spans as the index stores them: rounded to 3 decimals, (-1, -1) for the whole file."""
    if start is None or end is None or end < 0:
        return WHOLE, WHOLE
    return round(float(start), 3), round(float(end), 3)


def index_key(build_key: str, audio_sha256: str, span_start: float | None, span_end: float | None,
              params_hash: str) -> dict[str, Any]:
    s, e = span(span_start, span_end)
    return {"build_key": build_key, "audio_sha256": audio_sha256, "span_start": s, "span_end": e,
            "params_hash": params_hash}


class Lease:
    """A lease request. Heartbeats run from creation until release, while queued too."""

    def __init__(self, catalog: Catalog, data: dict[str, Any], heartbeat_s: float = HEARTBEAT_S):
        self._cat = catalog
        self.data = data
        self.lost: str | None = None  # why heartbeats stopped working, if they did
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, args=(heartbeat_s,), daemon=True,
                                        name=f"lease-{data['id']}")
        self._thread.start()

    id = property(lambda self: self.data["id"])
    holder = property(lambda self: self.data["holder"])
    models = property(lambda self: list(self.data["models"]))
    state = property(lambda self: self.data["state"])

    @property
    def active(self) -> bool:
        return self.state == "active" and self.lost is None

    def _beat(self, every: float) -> None:
        while not self._stop.wait(every):
            try:
                self.data = self._cat.heartbeat(self.id)
            except CatalogError as exc:
                if exc.status in (404, 409):  # expired or released: nothing to keep alive
                    self.lost = exc.message
                    return
            except httpx.HTTPError:
                pass  # the catalog is restarting; try again next beat (three misses are tolerated)

    def refresh(self) -> dict[str, Any]:
        self.data = self._cat.get_lease(self.id)
        return self.data

    def wait(self, timeout_s: float | None = None, on_wait: OnWait | None = None, poll_s: float = 2.0) -> Lease:
        """Block until the lease is active. on_wait(lease) is called whenever state, position or reason change."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        seen = None
        while True:
            d = self.data
            if d["state"] == "active":
                return self
            if d["state"] in ("failed", "expired", "released"):
                raise LeaseFailed(0, f"lease {d['id']} for {', '.join(d['models'])} {d['state']}: "
                                     f"{d.get('error') or d.get('waiting_for') or 'no reason given'}")
            if self.lost:
                raise LeaseFailed(0, f"lease {d['id']} lost: {self.lost}")
            sig = (d["state"], d.get("position"), d.get("waiting_for"))
            if on_wait and sig != seen:
                on_wait(d)
            seen = sig
            if deadline is not None and time.monotonic() >= deadline:
                raise LeaseFailed(0, f"lease {d['id']} not active after {timeout_s:.0f}s "
                                     f"({d['state']}, position {d.get('position')}: {d.get('waiting_for')})")
            time.sleep(poll_s)
            try:
                self.refresh()
            except httpx.HTTPError:
                pass

    def release(self) -> None:
        self._stop.set()
        try:
            self._cat.release(self.id)
        except (CatalogError, httpx.HTTPError):
            pass  # already gone, or the catalog is down: the lease expires on its own

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class Catalog:
    def __init__(self, url: str | None = None, *, http: httpx.Client | None = None, timeout: float = 30.0):
        self.url = (url or catalog_url() or DEFAULT_URL).rstrip("/")
        self._http = http or httpx.Client(base_url=self.url, timeout=timeout)

    # -------------------------------------------------------------------------- plumbing

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = self._http.request(method, "/api" + path, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
                body = body if isinstance(body, dict) else {}  # a JSON list, string or number has no fields
                err = body.get("error")
                msg = (err.get("message") if isinstance(err, dict) else err) or body.get("detail") or r.text
            except ValueError:
                msg = r.text
            raise CatalogError(r.status_code, str(msg)[:500])
        return r.json() if r.content else None

    def health(self) -> dict[str, Any]:
        return self._req("GET", "/health")

    def available(self) -> bool:
        try:
            self.health()
            return True
        except (CatalogError, httpx.HTTPError):
            return False

    # -------------------------------------------------------------------------- models

    def models(self, task: str | list[str] | None = None, tag: str | list[str] | None = None,
               state: str | None = None, all: bool = False) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = []
        for key, val in (("task", task), ("tag", tag)):
            for v in [val] if isinstance(val, str) else (val or []):
                params.append((key, v))
        if state:
            params.append(("state", state))
        if all:
            params.append(("all", "1"))
        return self._req("GET", "/models", params=params)

    def model(self, model_id: str) -> dict[str, Any]:
        return self._req("GET", f"/models/{model_id}")

    def register(self, model_id: str, *, display_name: str | None = None, tags: list[str] | None = None,
                 parent_id: str | None = None) -> dict[str, Any]:
        body = {k: v for k, v in {"display_name": display_name, "tags": tags, "parent_id": parent_id}.items()
                if v is not None}
        return self._req("POST", f"/models/{model_id}/register", json=body)

    def update(self, model_id: str, **fields: Any) -> dict[str, Any]:
        return self._req("PATCH", f"/models/{model_id}", json=fields)

    def unregister(self, model_id: str) -> None:
        self._req("DELETE", f"/models/{model_id}/register")

    def verify(self, model_id: str) -> dict[str, Any]:
        return self._req("POST", f"/models/{model_id}/verify")

    def start(self, model_id: str) -> dict[str, Any]:
        return self._req("POST", f"/models/{model_id}/start")

    def stop(self, model_id: str, force: bool = False) -> dict[str, Any]:
        return self._req("POST", f"/models/{model_id}/stop", params={"force": "1"} if force else None)

    def batch(self, model_id: str, *, vram_gb: float | None = None, item_s: float | None = None,
              item_tokens: int | None = None, n: int | None = None) -> dict[str, Any]:
        """The current build's batch profile: `recommendation` (the largest batch for vram_gb, default the full
        budget, and an item of item_s seconds or item_tokens tokens), `need` (GB that batch `n` needs) and `table`."""
        params = {k: v for k, v in {"vram_gb": vram_gb, "item_s": item_s, "item_tokens": item_tokens, "n": n}.items()
                  if v is not None}
        return self._req("GET", f"/models/{model_id}/batch", params=params or None)

    def batch_profile(self, model_id: str) -> dict[str, Any]:
        """Measure the model's batch profile (queued; it leases the model)."""
        return self._req("POST", f"/models/{model_id}/batch-profile")

    # -------------------------------------------------------------------------- leases

    def request_lease(self, holder: str, models: list[str], ttl_s: float = TTL_S) -> dict[str, Any]:
        return self._req("POST", "/leases", json={"holder": holder, "models": list(models), "ttl_s": ttl_s})

    def get_lease(self, lease_id: str) -> dict[str, Any]:
        return self._req("GET", f"/leases/{lease_id}")

    def heartbeat(self, lease_id: str) -> dict[str, Any]:
        return self._req("POST", f"/leases/{lease_id}/heartbeat")

    def release(self, lease_id: str) -> None:
        self._req("DELETE", f"/leases/{lease_id}")

    def leases(self) -> list[dict[str, Any]]:
        return self._req("GET", "/leases")

    def lease(self, holder: str, models: list[str], *, ttl_s: float = TTL_S, heartbeat_s: float = HEARTBEAT_S,
              timeout_s: float | None = None, on_wait: OnWait | None = None, poll_s: float = 2.0) -> Lease:
        """Request a lease and block until it is active. Use as a context manager to release it."""
        lease = Lease(self, self.request_lease(holder, models, ttl_s), heartbeat_s)
        try:
            return lease.wait(timeout_s, on_wait, poll_s)
        except BaseException:
            lease.release()
            raise

    # -------------------------------------------------------------------------- output index

    def index_lookup(self, app: str, keys: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        hits: list[dict[str, Any] | None] = []
        for i in range(0, len(keys), CHUNK):
            hits += self._req("POST", "/index/lookup", json={"app": app, "keys": keys[i:i + CHUNK]})["hits"]
        return hits

    def index_record(self, app: str, entries: list[dict[str, Any]]) -> int:
        n = 0
        for i in range(0, len(entries), CHUNK):
            n += self._req("POST", "/index/record", json={"app": app, "entries": entries[i:i + CHUNK]})["recorded"]
        return n

    def index_forget(self, app: str, output_ids: list[int | str]) -> int:
        return self._req("POST", "/index/forget", json={"app": app, "output_ids": list(output_ids)})["forgotten"]

    def register_app(self, name: str, base_url: str) -> dict[str, Any]:
        return self._req("POST", "/apps", json={"name": name, "base_url": base_url})

    def apps(self) -> list[dict[str, Any]]:
        return self._req("GET", "/apps")

    # -------------------------------------------------------------------------- planning, usage, events

    def plan(self, stages: list[dict[str, Any]], pipeline: str = "staged") -> dict[str, Any]:
        return self._req("POST", "/jobs", json={"pipeline": pipeline, "stages": stages})

    def usage(self) -> list[dict[str, Any]]:
        return self._req("GET", "/usage")

    def events(self, timeout: float | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        """(event, data) pairs from the server-sent event stream, until the connection closes."""
        with self._http.stream("GET", "/api/events", timeout=httpx.Timeout(10.0, read=timeout)) as r:
            event, data = "message", []
            for line in r.iter_lines():
                if not line:
                    if data:
                        yield event, json.loads("\n".join(data))
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())


class FakeCatalog:
    """In-memory Catalog for app tests: same methods, leases active at once, an in-process index.

    `models` are model records (see the README); each needs at least `id`, and usually `contract`, `api`,
    `build_key` and `capabilities`. `calls` records every method call as (name, args) for assertions.
    """

    url = "fake://catalog"

    def __init__(self, models: list[dict[str, Any]] | None = None):
        self._models = {m["id"]: dict(m) for m in (models or [])}
        self._leases: dict[str, dict[str, Any]] = {}
        self._index: dict[tuple, dict[str, Any]] = {}
        self._apps: dict[str, dict[str, Any]] = {}
        self._n = 0
        self.calls: list[tuple[str, tuple]] = []

    def _log(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def add_model(self, record: dict[str, Any]) -> None:
        self._models[record["id"]] = dict(record)

    def batch(self, model_id: str, *, vram_gb: float | None = None, item_s: float | None = None,
              item_tokens: int | None = None, n: int | None = None) -> dict[str, Any]:
        """A record's `batch` (e.g. {"recommendation": {"batch": 8}}), else no profile."""
        self._log("batch", model_id, vram_gb, item_s, item_tokens, n)
        return {"model": model_id, "status": "none", "measuring": None, **(self._models.get(model_id, {}).get("batch") or {})}

    def batch_profile(self, model_id: str) -> dict[str, Any]:
        self._log("batch_profile", model_id)
        return {"model": model_id, "measuring": "pending"}

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def available(self) -> bool:
        return True

    def models(self, task=None, tag=None, state=None, all=False) -> list[dict[str, Any]]:  # noqa: A002
        tasks = [task] if isinstance(task, str) else list(task or [])
        tags = [tag] if isinstance(tag, str) else list(tag or [])
        out = []
        for m in self._models.values():
            if not all and not m.get("registered", True):
                continue
            seen = set((m.get("capabilities") or {}).get("tasks") or []) | set(m.get("tasks") or [])
            if tasks and not seen & set(tasks):
                continue
            if tags and not set(tags) <= set(m.get("tags") or []):
                continue
            if state and m.get("state", "stopped") != state:
                continue
            out.append(dict(m))
        return out

    def model(self, model_id: str) -> dict[str, Any]:
        for m in self._models.values():
            if model_id == m["id"] or model_id in (m.get("aliases") or []):
                return dict(m)
        raise CatalogError(404, f"unknown model {model_id!r}")

    def register(self, model_id: str, *, display_name=None, tags=None, parent_id=None) -> dict[str, Any]:
        self._log("register", model_id)
        m = self._models.setdefault(model_id, {"id": model_id})
        m["registered"] = True
        m.update({k: v for k, v in {"display_name": display_name, "tags": tags, "parent_id": parent_id}.items()
                  if v is not None})
        return dict(m)

    def update(self, model_id: str, **fields: Any) -> dict[str, Any]:
        self._models[model_id].update(fields)
        return dict(self._models[model_id])

    def unregister(self, model_id: str) -> None:
        self._models[model_id]["registered"] = False

    def verify(self, model_id: str) -> dict[str, Any]:
        self._log("verify", model_id)
        return self._models[model_id].get("capabilities") or {"status": "pending"}

    def start(self, model_id: str) -> dict[str, Any]:
        self._models[model_id]["state"] = "running"
        return {"model": dict(self._models[model_id]), "request": None}

    def stop(self, model_id: str, force: bool = False) -> dict[str, Any]:
        self._models[model_id]["state"] = "stopped"
        return dict(self._models[model_id])

    def request_lease(self, holder: str, models: list[str], ttl_s: float = TTL_S) -> dict[str, Any]:
        self._n += 1
        lease = {"id": f"l-{self._n}", "holder": holder, "models": list(models), "state": "active",
                 "position": None, "waiting_for": None, "error": None}
        self._leases[lease["id"]] = lease
        self._log("lease", holder, tuple(models))
        return dict(lease)

    def get_lease(self, lease_id: str) -> dict[str, Any]:
        return dict(self._leases[lease_id])

    def heartbeat(self, lease_id: str) -> dict[str, Any]:
        return dict(self._leases[lease_id])

    def release(self, lease_id: str) -> None:
        self._leases[lease_id]["state"] = "released"
        self._log("release", lease_id)

    def leases(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._leases.values() if x["state"] in ("queued", "starting", "active")]

    def lease(self, holder: str, models: list[str], **_: Any) -> Lease:
        return Lease(self, self.request_lease(holder, models), heartbeat_s=3600)  # type: ignore[arg-type]

    @staticmethod
    def _key(k: dict[str, Any]) -> tuple:
        s, e = span(k["span_start"], k["span_end"])
        return k["build_key"], k["audio_sha256"], s, e, k["params_hash"]

    def index_lookup(self, app: str, keys: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        self._log("index_lookup", app, len(keys))
        by_key: dict[tuple, list[dict[str, Any]]] = {}  # one pass over the index, in the order entries were recorded
        for (key, _), v in self._index.items():
            by_key.setdefault(key, []).append(v)
        out = []
        for k in keys:
            owners = by_key.get(self._key(k), [])
            mine = [v for v in owners if v["owner"] == app]
            hit = (mine or owners or [None])[0]
            out.append(dict(hit) if hit else None)
        return out

    def index_record(self, app: str, entries: list[dict[str, Any]]) -> int:
        self._log("index_record", app, len(entries))
        for e in entries:
            self._index[(self._key(e), app)] = {"owner": app, "output_id": e["output_id"],
                                                "owner_url": (self._apps.get(app) or {}).get("base_url")}
        return len(entries)

    def index_forget(self, app: str, output_ids: list[int | str]) -> int:
        drop = [k for k, v in self._index.items() if v["owner"] == app and v["output_id"] in set(output_ids)]
        for k in drop:
            del self._index[k]
        return len(drop)

    def register_app(self, name: str, base_url: str) -> dict[str, Any]:
        self._apps[name] = {"name": name, "base_url": base_url}
        for v in self._index.values():
            if v["owner"] == name:
                v["owner_url"] = base_url
        return dict(self._apps[name])

    def apps(self) -> list[dict[str, Any]]:
        return list(self._apps.values())

    def plan(self, stages: list[dict[str, Any]], pipeline: str = "staged") -> dict[str, Any]:
        self._log("plan", pipeline, len(stages))
        steps, loads, hits_total = [], 0, 0
        for st in stages:
            for mid in st["models"]:
                m = self._models.get(mid, {})
                keys = (st.get("keys") or {}).get(mid)
                hits = None
                if keys is not None:
                    bk = m.get("build_key")
                    found = self.index_lookup("plan", [{**k, "build_key": bk} for k in keys]) if bk else []
                    hits = sum(1 for h in found if h)
                    hits_total += hits
                missing = None if keys is None else len(keys) - (hits or 0)
                skip = missing == 0
                load = not skip and m.get("state") != "running"
                loads += load
                steps.append({"stage": st["name"], "model": mid, "load": load,
                              "cold_start_s": m.get("cold_start_s", 30.0), "items": None if keys is None else len(keys),
                              "hits": hits, "missing": missing, "skip": skip})
        return {"pipeline": pipeline, "steps": steps, "loads": loads,
                "est_load_s": sum(s["cold_start_s"] for s in steps if s["load"]), "hits": hits_total,
                "fits_together": True}

    def usage(self) -> list[dict[str, Any]]:
        return []

    def events(self, timeout: float | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(())
