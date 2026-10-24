"""Scheduler: which model services run, who holds them, and who waits.

Requests (leases from apps, starts asked for by the gateway or `just up`) wait in one strict
first-come-first-served queue. The head of the queue is granted when its models fit in GPU memory,
after stopping, if needed, the unleased models that are cheapest to bring back. A request behind a
blocked head is granted only if it needs no new memory and protects nothing the head could use.

Rules (README "Scheduler rules"): fit by declared footprints; never stop a leased model, a model with
calls in flight, or one that has not run for its settle time (2 x cold start, at least 60 s); stop the
lowest reload cost first (cold start x likelihood of reuse); stop only enough. One model starts at a
time, on the starter thread, which is also the only thread that stops models.

Every model loads entirely on the GPU: besides the declared fit, a GPU model starts only when the device
(measured, so other processes count) has its whole budget free, and not while another GPU model is still
loading (vLLM sizes its KV cache from memory that must not change under it). A model a lease brought
up is stopped once that lease ends and no live or queued request needs it.

Hosted endpoints (catalog/hosted.py) may be named in a lease next to platform models. They use no GPU and
have no container, so they are granted at once and never wait in the queue.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools import render
from tools.schema import Manifest, Platform, PlatformError, load_platform

from .builds import Build, Builds
from .db import DB, dumps, iso, now_iso
from .events import Events
from .runtime import ContainerInfo, Runtime, StartError, budget_gib
from .settings import Settings

log = logging.getLogger("catalog.scheduler")
LIVE = ("queued", "starting", "active")
EPS = 1e-9


class SchedulerError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Request:
    id: str
    seq: int
    kind: str  # lease (an app's job) | start (gateway or `just up`: ends once the models are up)
    holder: str
    models: list[str]
    state: str  # queued | starting | active | released | expired | failed | done
    ttl_s: float
    created_at: float
    heartbeat_at: float
    granted_at: float | None = None
    ready_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    waiting_for: str | None = None
    position: int | None = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "holder": self.holder, "models": list(self.models),
                "state": self.state, "position": self.position, "waiting_for": self.waiting_for, "error": self.error,
                "created_at": iso(self.created_at), "granted_at": iso(self.granted_at), "ready_at": iso(self.ready_at),
                "ended_at": iso(self.ended_at),
                "expires_at": iso(self.heartbeat_at + self.ttl_s) if self.state in LIVE else None}


@dataclass
class Run:
    """What the scheduler knows about one service's container."""

    running: bool = False
    ready_at: float | None = None  # when it became healthy (or was first seen running)
    starting: bool = False
    start_began: float | None = None
    stopping: bool = False
    failed: str | None = None  # the last start's failure, until it runs again
    in_flight: int = 0
    calls: int = 0
    errors: int = 0
    last_used: float | None = None
    container: ContainerInfo | None = None
    build: Build | None = None
    usage_dirty: bool = False
    pinned: bool = False  # started on purpose (manual start, `just up`, outside the catalog): never spun down
    released_at: float | None = None  # when the last lease holding it ended: spun down once nothing needs it
    short: str | None = None  # the start found too little free GPU memory: its request queues again


@dataclass
class Fit:
    ok: bool
    evict: list[str] = field(default_factory=list)
    reason: str | None = None
    fatal: bool = False  # can never be met: fail the request


class Scheduler:
    def __init__(self, db: DB, settings: Settings, runtime: Runtime, builds: Builds, events: Events):
        self.db = db
        self.settings = settings
        self.runtime = runtime
        self.builds = builds
        self.events = events
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.requests: dict[str, Request] = {}
        self.runs: dict[str, Run] = {}
        self._seq = 0
        self._jobs: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._platform: Platform | None = None
        self._platform_sig: tuple = ()
        self.platform_error: str | None = None
        self._last_published: dict[str, Any] = {}
        self._cold_cache: dict[str, list[float]] = {}
        self.gpu: tuple[float, float] | None = None  # (total, free) GiB at the last tick; None: not measurable
        self.hosted_ids: Callable[[], set[str]] = set  # set by the catalog: ids of hosted endpoints
        self._load_requests()

    # ------------------------------------------------------------------------- platform

    def _manifest_sig(self) -> tuple:
        root = self.settings.platform_dir
        files = [root / "config.yaml", root / "engines" / "engines.yaml", *sorted((root / "services").glob("*.y*ml"))]
        return tuple((str(f), f.stat().st_mtime_ns) for f in files if f.exists())

    def platform(self) -> Platform:
        sig = self._manifest_sig()
        with self.lock:
            if self._platform is not None and sig == self._platform_sig:
                return self._platform
        try:
            p = load_platform(self.settings.platform_dir)
        except PlatformError as exc:
            with self.lock:
                self.platform_error = "; ".join(exc.errors)
                if self._platform is None:
                    raise
                return self._platform
        if self.settings.docker:
            render.write_outputs(p)  # compose.yml must match the manifests the catalog starts from
        with self.lock:
            self._platform, self._platform_sig, self.platform_error = p, sig, None
            for s in p.services:
                self.runs.setdefault(s.name, Run())
        return p

    def manifest(self, name: str) -> Manifest:
        try:
            return self.platform().get(name)
        except KeyError as exc:
            raise SchedulerError(404, exc.args[0]) from None

    # ------------------------------------------------------------------------- persistence

    def _load_requests(self) -> None:
        now = time.time()
        for row in self.db.all("SELECT * FROM request WHERE state IN ('queued', 'starting', 'active') ORDER BY seq"):
            r = Request(row["id"], row["seq"], row["kind"], row["holder"], json.loads(row["models"]),
                        row["state"], row["ttl_s"], row["created_at"], now, row["granted_at"], row["ready_at"])
            if r.state == "starting":
                r.state = "queued"  # re-evaluated on the first tick
            self.requests[r.id] = r
        top = self.db.one("SELECT MAX(seq) AS m FROM request")
        self._seq = (top or {}).get("m") or 0

    def _save(self, r: Request) -> None:
        self.db.execute(
            "INSERT INTO request (id, seq, kind, holder, models, state, ttl_s, created_at, granted_at, ready_at, "
            "heartbeat_at, ended_at, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "state = excluded.state, granted_at = excluded.granted_at, ready_at = excluded.ready_at, "
            "heartbeat_at = excluded.heartbeat_at, ended_at = excluded.ended_at, error = excluded.error",
            (r.id, r.seq, r.kind, r.holder, dumps(r.models), r.state, r.ttl_s, r.created_at, r.granted_at, r.ready_at,
             r.heartbeat_at, r.ended_at, r.error))

    # ------------------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.platform()
        self._threads = [threading.Thread(target=self._loop, daemon=True, name="scheduler"),
                         threading.Thread(target=self._starter, daemon=True, name="starter")]
        for t in self._threads:
            t.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._jobs.put(("quit", None))
        with self.changed:
            self.changed.notify_all()
        for t in self._threads:
            t.join(timeout=5)

    def wake(self) -> None:
        with self.changed:
            self.changed.notify_all()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick must not stop scheduling
                log.exception("scheduler tick failed")
            with self.changed:
                self.changed.wait(timeout=self.settings.reconcile_s)

    # ------------------------------------------------------------------------- the tick

    def tick(self) -> None:
        p = self.platform()
        try:
            containers = self.runtime.containers(p)
        except Exception as exc:  # noqa: BLE001 - docker hiccup: keep the last view
            log.warning("cannot list containers: %s", exc)
            containers = None
        builds = {}
        for s in p.services:  # build keys need docker and the hasher: outside the lock
            c = (containers or {}).get(s.name)
            try:
                builds[s.name] = self.builds.current(p, s, c)
            except Exception as exc:  # noqa: BLE001
                log.warning("build key of %s: %s", s.name, exc)
        try:
            gpu = self.runtime.gpu_memory(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read GPU memory: %s", exc)
            gpu = None
        now = time.time()
        with self.lock:
            self.gpu = gpu
            if containers is not None:
                self._reconcile(p, containers, now)
            for name, b in builds.items():
                self.runs[name].build = b
            self._expire(now)
            self._drop_unknown(p, now)
            self._spin_down(now)
            self._schedule(p, now)
            self._promote(p, now)
            self._positions()
            for run in self.runs.values():
                run.short = None  # every request it held has been queued again
            self._flush_usage()
            self._publish(p)

    def _reconcile(self, p: Platform, containers: dict[str, ContainerInfo], now: float) -> None:
        for name in set(self.runs) | set(containers):
            run = self.runs.setdefault(name, Run())
            c = containers.get(name)
            run.container = c
            if run.starting or run.stopping:
                continue  # the starter thread owns these until it finishes
            up = c is not None and c.state == "running" and c.health in ("healthy", "-")
            # Docker's health check lags the HTTP one the starter waits on: "starting" is not "down".
            alive = c is not None and c.state == "running" and c.health != "unhealthy"
            if up and not run.running:
                run.running, run.ready_at, run.failed = True, now, None
                # Not started by the starter: outside the catalog, or before a catalog restart. Keep it up
                # unless a live lease (reloaded from the database) holds it.
                run.pinned = not any(name in r.models for r in self._live() if r.kind == "lease")
            elif not alive and run.running:
                run.running, run.ready_at = False, None
                run.pinned, run.released_at = False, None

    def _expire(self, now: float) -> None:
        for r in self.requests.values():
            if r.state in LIVE and now - r.heartbeat_at > r.ttl_s:
                self._end(r, "expired", now, f"no heartbeat for {r.ttl_s:.0f}s")

    def _drop_unknown(self, p: Platform, now: float) -> None:
        known = {s.name for s in p.services} | self.hosted_ids()
        for r in self.requests.values():
            missing = [m for m in r.models if m not in known]
            if r.state in LIVE and missing:
                self._end(r, "failed", now, f"no longer defined on the platform: {', '.join(missing)}")

    def _end(self, r: Request, state: str, now: float, error: str | None = None) -> None:
        r.state, r.ended_at, r.position = state, now, None
        if error:
            r.error = error
        if state != "done":
            r.waiting_for = None
        if r.kind == "lease":
            for m in self._local(r):
                self.runs.setdefault(m, Run()).released_at = now
        self._save(r)
        self.changed.notify_all()

    def _spin_down(self, now: float) -> None:
        """Stop unpinned models no live or queued request needs: after the linger once their lease ended, or after
        the idle timeout when only gateway calls without a lease started and used them."""
        wanted = {m for r in self._live() for m in r.models}
        for name, run in self.runs.items():
            if run.pinned or name in wanted or not run.running or run.starting or run.stopping or run.in_flight:
                continue
            if run.released_at is not None:
                idle, limit, why = now - max(run.released_at, run.last_used or 0), self.settings.release_linger_s, \
                    "its lease ended and no request needs it"
            else:
                idle, limit, why = now - max(run.ready_at or now, run.last_used or 0), self.settings.gateway_idle_s, \
                    f"no call for {self.settings.gateway_idle_s:.0f}s since the gateway started it"
            if idle >= limit:
                run.stopping = True
                self._jobs.put(("stop", name))
                log.info("stopping %s: %s", name, why)

    def _live(self) -> list[Request]:
        return sorted((r for r in self.requests.values() if r.state in LIVE), key=lambda r: r.seq)

    def _protected(self) -> dict[str, list[str]]:
        """Models held by granted requests -> holders. Leased models are never stopped."""
        out: dict[str, list[str]] = {}
        for r in self._live():
            if r.state in ("starting", "active") and r.kind == "lease":
                for m in r.models:
                    out.setdefault(m, []).append(r.holder)
            elif r.state == "starting":  # a start request protects its models until they are up
                for m in r.models:
                    out.setdefault(m, []).append(r.holder)
        return out

    def _schedule(self, p: Platform, now: float) -> None:
        known = {s.name: s for s in p.services}
        protected = self._protected()
        head: Request | None = None
        for r in self._live():
            if r.state != "queued":
                continue
            gpu = [m for m in r.models if m in known and known[m].gpu]
            if head is not None:
                # Behind a blocked head: only requests that take no memory and protect nothing new may pass.
                passes = all(m in protected and self.runs[m].running for m in gpu)
                if passes:
                    self._grant(p, r, [], now)
                    for m in r.models:
                        protected.setdefault(m, []).append(r.holder)
                else:
                    r.waiting_for = f"queued behind {head.holder} ({head.waiting_for})"
                continue
            fit = self._fit(p, r, protected, now)
            if fit.fatal:
                self._end(r, "failed", now, fit.reason)
                continue
            if fit.ok:
                self._grant(p, r, fit.evict, now)
                for m in r.models:
                    protected.setdefault(m, []).append(r.holder)
            else:
                r.waiting_for = fit.reason
                head = r

    def frac(self, p: Platform, name: str) -> float:
        try:
            return p.get(name).vram_fraction(p.config.gpu_total_gb)
        except KeyError:
            return 0.0

    def _occupying(self, p: Platform) -> set[str]:
        """GPU models that hold (or are about to hold) memory: running or starting, and not being stopped."""
        return {n for n, run in self.runs.items()
                if (run.running or run.starting) and not run.stopping and self.frac(p, n) > 0}

    def _fit(self, p: Platform, r: Request, protected: dict[str, list[str]], now: float) -> Fit:
        fit = self._fit_declared(p, r, protected, now)
        if not fit.ok or self.gpu is None:
            return fit
        return self._fit_measured(p, r, protected, now, fit)

    def gpu_gib(self, p: Platform, name: str, total_gib: float) -> float:
        try:
            return budget_gib(p.get(name), total_gib)
        except KeyError:
            return 0.0

    def _fit_measured(self, p: Platform, r: Request, protected: dict[str, list[str]], now: float, fit: Fit) -> Fit:
        """The models to load must find their whole budget free on the device, so none is partly loaded or
        sized from memory another process is still taking. Stops more idle models if declared footprints
        were not enough (other processes, CUDA contexts), else waits."""
        assert self.gpu is not None
        total, free = self.gpu
        load = [m for m in r.models if self.frac(p, m) > 0 and not self.runs[m].running and not self.runs[m].starting]
        if not load:
            return fit
        loading = sorted(m for m, run in self.runs.items() if run.starting and m not in r.models and self.frac(p, m) > 0)
        if loading:
            return Fit(False, reason=f"waiting for {', '.join(loading)} to finish loading before measuring free GPU memory")
        need = sum(self.gpu_gib(p, m, total) for m in load) + self.settings.gpu_margin_gib
        evict = list(fit.evict)
        # Models being stopped give back about their budget (custom services hold less; the start checks again).
        free += sum(self.gpu_gib(p, m, total) for m, run in self.runs.items() if run.stopping or m in evict)
        candidates = [m for m in self._occupying(p) - set(r.models) - set(protected) - set(evict)
                      if self._evictable(p, m, now)]
        candidates.sort(key=lambda m: (self.reload_cost(p, m, now), self.runs[m].last_used or 0, m))
        for m in candidates:
            if free >= need:
                break
            evict.append(m)
            free += self.gpu_gib(p, m, total)
        if free >= need:
            return Fit(True, evict)
        platform = sum(self.gpu_gib(p, m, total) for m in self._occupying(p) | {m for m, run in self.runs.items() if run.stopping})
        other = total - self.gpu[1] - platform
        return Fit(False, reason=f"{', '.join(sorted(load))} need {need:.1f} GiB free on the GPU to load entirely; "
                                 f"{free:.1f} GiB would be free" + (f" (about {other:.1f} GiB is used outside the platform's "
                                                                    "models)" if other > 0.5 else ""))

    def _fit_declared(self, p: Platform, r: Request, protected: dict[str, list[str]], now: float) -> Fit:
        limit = p.config.util_limit
        need = {m for m in r.models if self.frac(p, m) > 0}
        alone = sum(self.frac(p, m) for m in need)
        if alone > limit + EPS:
            return Fit(False, reason=f"{', '.join(sorted(need))} need {alone:.2f} of the GPU together, over util_limit "
                                     f"{limit}; they can never run at once", fatal=True)
        held = {m for m in protected if self.frac(p, m) > 0}
        occupied = self._occupying(p) | held | need
        used = sum(self.frac(p, m) for m in occupied)
        if used <= limit + EPS:
            return Fit(True)
        candidates = [m for m in occupied - need - set(protected) if self._evictable(p, m, now)]
        candidates.sort(key=lambda m: (self.reload_cost(p, m, now), self.runs[m].last_used or 0, m))
        evict = []
        for m in candidates:
            if used <= limit + EPS:
                break
            evict.append(m)
            used -= self.frac(p, m)
        if used <= limit + EPS:
            return Fit(True, evict)
        # Explain why not, and fail if waiting cannot help (the holder blocks itself).
        own = {m for m, hs in protected.items() if r.holder in hs and self.frac(p, m) > 0}
        others_held = {m for m, hs in protected.items() if any(h != r.holder for h in hs) and self.frac(p, m) > 0}
        if own - others_held:
            floor = sum(self.frac(p, m) for m in (others_held | need))
            if floor <= limit + EPS and floor + sum(self.frac(p, m) for m in own - others_held - need) > limit + EPS:
                return Fit(False, fatal=True, reason=f"{r.holder} already leases {', '.join(sorted(own - need))}; "
                                                     "release that lease before asking for more")
        blockers = []
        for m in sorted(occupied - need):
            run = self.runs[m]
            if m in protected:
                blockers.append(f"{m} (leased by {' and '.join(sorted(set(protected[m])))})")
            elif run.in_flight:
                blockers.append(f"{m} ({run.in_flight} call(s) in flight)")
            elif run.starting:
                blockers.append(f"{m} (starting)")
            elif not self._settled(p, m, now) and not self._unneeded(m):
                left = self.settle_s(p, m) - (now - (run.ready_at or now))
                blockers.append(f"{m} (settling, {max(0, left):.0f}s left)")
        return Fit(False, reason=f"{', '.join(sorted(need))} does not fit next to " + (", ".join(blockers) or "running models"))

    def _evictable(self, p: Platform, m: str, now: float) -> bool:
        run = self.runs[m]
        return (run.running and not run.starting and not run.stopping and run.in_flight == 0
                and (self._settled(p, m, now) or self._unneeded(m)))

    def _unneeded(self, m: str) -> bool:
        """Its lease ended and nobody started it on purpose: no settle time to wait out before stopping it."""
        run = self.runs[m]
        return run.released_at is not None and not run.pinned

    def settle_s(self, p: Platform, m: str) -> float:
        return max(self.settings.settle_min_s, 2 * self.cold_start(p, m)[0])

    def _settled(self, p: Platform, m: str, now: float) -> bool:
        run = self.runs[m]
        return run.ready_at is not None and now - run.ready_at >= self.settle_s(p, m)

    def reload_cost(self, p: Platform, m: str, now: float) -> float:
        """Cold start x likelihood of reuse: queued demand plus recent use (decaying over reuse_tau_s)."""
        queued = sum(1 for r in self._live() if r.state == "queued" and m in r.models)
        run = self.runs[m]
        last = run.last_used or run.ready_at or 0
        recency = math.exp(-(now - last) / self.settings.reuse_tau_s) if last else 0.0
        return self.cold_start(p, m)[0] * (min(1, queued) + recency)

    def _grant(self, p: Platform, r: Request, evict: list[str], now: float) -> None:
        r.state, r.granted_at, r.waiting_for = "starting", now, None
        for m in evict:
            self.runs[m].stopping = True
            self._jobs.put(("stop", m))
            log.info("stopping %s to make room for %s", m, r.holder)
        for m in self._local(r):
            run = self.runs[m]
            if not run.running and not run.starting:
                run.starting, run.failed = True, None
                self._jobs.put(("start", m))
        self._save(r)

    def _local(self, r: Request) -> list[str]:
        """The request's platform models: hosted endpoints have nothing to start or watch."""
        hosted = self.hosted_ids()
        return [m for m in r.models if m not in hosted]

    def _promote(self, p: Platform, now: float) -> None:
        """Granted requests become active once every model is up; active leases restart crashed models."""
        for r in self._live():
            if r.state == "queued":
                continue
            local = self._local(r)
            short = [m for m in local if self.runs[m].short and not self.runs[m].running and not self.runs[m].starting]
            if short:
                # Not loaded at all rather than partly: back in its queue place, fitted again on the next tick.
                r.state, r.granted_at, r.waiting_for = "queued", None, "; ".join(self.runs[m].short for m in short)
                self._save(r)
                continue
            failed = [m for m in local if self.runs[m].failed and not self.runs[m].running and not self.runs[m].starting]
            if failed and r.state == "starting":
                self._end(r, "failed", now, "; ".join(f"{m}: {self.runs[m].failed}" for m in failed))
                continue
            down = [m for m in local if not self.runs[m].running]
            # Active means callable *and* identified: outputs made under a lease can always be indexed.
            unkeyed = [m for m in local if not (self.runs[m].build and self.runs[m].build.build_key)]
            if not down and unkeyed and r.state == "starting":
                r.waiting_for = "; ".join(f"{m}: {self.runs[m].build.pending if self.runs[m].build else 'build not inspected yet'}"
                                          for m in unkeyed)
                continue
            if not down and r.state == "starting":
                r.ready_at = now
                if r.kind == "start":
                    self._end(r, "done", now)
                else:
                    r.state = "active"
                    self._save(r)
            elif down and r.state == "active":
                for m in down:  # a leased model crashed or was stopped by hand: bring it back
                    run = self.runs[m]
                    if not run.starting:
                        run.starting, run.failed = True, None
                        self._jobs.put(("start", m))
                r.state = "starting"
                self._save(r)

    def _positions(self) -> None:
        n = 0
        for r in self._live():
            if r.state == "queued":
                n += 1
                r.position = n
            else:
                r.position = None

    # ------------------------------------------------------------------------- starter thread

    def _starter(self) -> None:
        while not self._stop.is_set():
            kind, name = self._jobs.get()
            if kind == "quit":
                return
            try:
                if kind == "stop":
                    self._do_stop(name)
                elif kind == "start":
                    self._do_start(name)
            except Exception:  # noqa: BLE001
                log.exception("%s %s failed", kind, name)
            self.wake()

    def _do_stop(self, name: str) -> None:
        p = self.platform()
        try:
            self.runtime.stop(p, [name])
        finally:
            with self.lock:
                run = self.runs[name]
                run.stopping, run.running, run.ready_at = False, False, None
                run.pinned, run.released_at = False, None

    def _do_start(self, name: str) -> None:
        p = self.platform()
        with self.lock:
            run = self.runs[name]
            if run.running:
                run.starting = False
                return
            run.start_began = time.time()
        s = p.get(name)
        short = self._wait_for_gpu(p, s)
        if short:
            log.warning("not starting %s: %s", name, short)
            with self.lock:
                run.starting, run.short = False, short
            return
        url = self.upstream_url(name)
        t0 = time.monotonic()
        error = None
        try:
            log.info("starting %s", name)
            self.runtime.start(p, s)
            while True:
                if self.runtime.healthy(p, s, url):
                    break
                c = None
                try:
                    c = self.runtime.containers(p).get(name)
                except Exception:  # noqa: BLE001
                    pass
                if c is not None and (c.state in ("exited", "dead") or c.health == "unhealthy"):
                    error = f"container {c.state}/{c.health}"
                    break
                if time.monotonic() - t0 > self.settings.start_timeout_s:
                    error = f"not healthy after {self.settings.start_timeout_s:.0f}s"
                    break
                if self._stop.wait(1.0):
                    return
        except StartError as exc:
            error = str(exc)
        seconds = time.monotonic() - t0
        if error:
            detail = ""
            try:
                detail = self.runtime.failure_detail(p, name)
            except Exception:  # noqa: BLE001
                pass
            log.warning("start of %s failed: %s", name, error)
            with self.lock:
                run.starting, run.running, run.failed = False, False, error + (f"\n{detail}" if detail else "")
            if self.settings.docker:
                self.runtime.stop(p, [name])
            return
        build = None
        try:
            c = self.runtime.containers(p).get(name)
            build = self.builds.current(p, s, c)
        except Exception:  # noqa: BLE001
            pass
        self.db.execute("INSERT INTO cold_start (model_id, build_key, seconds, at) VALUES (?, ?, ?, ?)",
                        (name, build.build_key if build else None, seconds, now_iso()))
        with self.lock:
            self._cold_cache.clear()
            run.starting, run.running, run.ready_at, run.failed = False, True, time.time(), None
            if build:
                run.build = build
        log.info("%s healthy after %.1fs", name, seconds)

    def _wait_for_gpu(self, p: Platform, s: Manifest) -> str | None:
        """Wait until the device has the model's whole budget free (stopped models give memory back a moment
        after they exit). None when it has, or memory cannot be measured; else why it was not started."""
        if not s.gpu:
            return None
        deadline = time.monotonic() + self.settings.gpu_free_wait_s
        while True:
            try:
                gpu = self.runtime.gpu_memory(p)
            except Exception:  # noqa: BLE001
                gpu = None
            if gpu is None:
                return None
            need = budget_gib(s, gpu[0]) + self.settings.gpu_margin_gib
            if gpu[1] >= need:
                return None
            if time.monotonic() >= deadline or self._stop.wait(1.0):
                return f"{s.name} needs {need:.1f} GiB free on the GPU to load entirely; {gpu[1]:.1f} GiB free"

    # ------------------------------------------------------------------------- timings

    def cold_start(self, p: Platform, name: str) -> tuple[float, bool]:
        """(seconds, measured): median of this build's loads, else the model's, else an estimate."""
        run = self.runs.get(name)
        bk = run.build.build_key if run and run.build else None
        key = f"{name}|{bk}"
        with self.lock:
            cached = self._cold_cache.get(key)
        if cached is None:
            rows = [r["seconds"] for r in self.db.all("SELECT seconds FROM cold_start WHERE model_id = ? AND build_key IS ? "
                                                      "ORDER BY id DESC LIMIT 15", (name, bk))]
            if not rows:
                rows = [r["seconds"] for r in self.db.all("SELECT seconds FROM cold_start WHERE model_id = ? "
                                                          "ORDER BY id DESC LIMIT 15", (name,))]
            cached = rows
            with self.lock:
                self._cold_cache[key] = rows
        if cached:
            return statistics.median(cached), True
        try:
            s = p.get(name)
        except KeyError:
            return self.settings.estimate_engine_s, False
        if not s.gpu:
            return self.settings.estimate_cpu_s, False
        return (self.settings.estimate_custom_gpu_s if s.is_custom else self.settings.estimate_engine_s), False

    # ------------------------------------------------------------------------- requests (API)

    def request(self, holder: str, models: list[str], kind: str = "lease", ttl_s: float | None = None) -> Request:
        if not holder.strip():
            raise SchedulerError(422, "holder must not be empty")
        if not models:
            raise SchedulerError(422, "models must not be empty")
        p = self.platform()
        hosted = self.hosted_ids()
        try:
            local = iter(s.name for s in p.resolve([m for m in models if m not in hosted]))
            names = [m if m in hosted else next(local) for m in models]
        except KeyError as exc:
            raise SchedulerError(404, exc.args[0]) from None
        now = time.time()
        with self.lock:
            self._seq += 1
            r = Request(f"l-{uuid.uuid4().hex[:12]}", self._seq, kind, holder.strip(), names, "queued",
                        float(ttl_s or self.settings.lease_ttl_s), now, now)
            if kind == "start" and r.holder != "gateway":
                self._pin(self._local(r))
            self.requests[r.id] = r
            self._save(r)
            self._schedule(p, now)
            self._promote(p, now)
            self._positions()
            self.changed.notify_all()
            return r

    def get(self, rid: str) -> Request:
        with self.lock:
            r = self.requests.get(rid)
        if r is None:
            row = self.db.one("SELECT * FROM request WHERE id = ?", (rid,))
            if row is None:
                raise SchedulerError(404, f"unknown lease {rid!r}")
            return Request(row["id"], row["seq"], row["kind"], row["holder"], json.loads(row["models"]), row["state"],
                           row["ttl_s"], row["created_at"], row["heartbeat_at"], row["granted_at"], row["ready_at"],
                           row["ended_at"], row["error"])
        return r

    def heartbeat(self, rid: str) -> Request:
        with self.lock:
            r = self.get(rid)
            if r.state not in LIVE:
                raise SchedulerError(409, f"lease {rid} is {r.state}" + (f": {r.error}" if r.error else ""))
            r.heartbeat_at = time.time()
            self._save(r)
            return r

    def release(self, rid: str) -> Request:
        with self.lock:
            r = self.get(rid)
            if r.state in LIVE:
                self._end(r, "released", time.time())
            return r

    def live_requests(self) -> list[Request]:
        with self.lock:
            return self._live()

    # ------------------------------------------------------------------------- gateway, manual start/stop

    def upstream_url(self, name: str) -> str:
        from aicore.endpoints import resolve_base

        p = self.platform()
        s = p.get(name)
        entry = {"name": s.name, "host_port": s.port, "internal_port": p.internal_port(s)}
        return resolve_base(entry, self.settings.upstream_host)

    def ensure_started(self, name: str, holder: str = "gateway") -> Request | None:
        """A start request for the model unless it is up or a live request already covers it."""
        if name in self.hosted_ids():
            raise SchedulerError(409, f"{name} is a hosted endpoint: it is always up and cannot be started")
        with self.lock:
            run = self.runs.get(name) or Run()
            if run.running and not run.stopping:
                if holder != "gateway":
                    self._pin([name])
                return None
            for r in self._live():
                if name in r.models:
                    if holder != "gateway":
                        self._pin([name])
                    return r
        return self.request(holder, [name], kind="start")

    def _pin(self, names: list[str]) -> None:
        """Started on purpose (not a gateway call): kept up after leases end, until evicted or stopped."""
        for m in names:
            self.runs.setdefault(m, Run()).pinned = True

    def begin_call(self, name: str) -> None:
        with self.lock:
            run = self.runs.setdefault(name, Run())
            run.in_flight += 1
            run.last_used = time.time()

    def end_call(self, name: str, error: bool) -> None:
        with self.lock:
            run = self.runs.setdefault(name, Run())
            run.in_flight = max(0, run.in_flight - 1)
            run.calls += 1
            run.errors += int(error)
            run.last_used = time.time()
            run.usage_dirty = True

    def is_up(self, name: str) -> bool:
        with self.lock:
            run = self.runs.get(name)
            return bool(run and run.running and not run.stopping)

    def stop_model(self, name: str, force: bool = False) -> None:
        if name in self.hosted_ids():
            raise SchedulerError(409, f"{name} is a hosted endpoint: it has no container to stop")
        with self.lock:
            holders = self._protected().get(name, [])
            if holders and not force:
                raise SchedulerError(409, f"{name} is leased by {', '.join(sorted(set(holders)))}; force=1 overrides")
            run = self.runs.setdefault(name, Run())
            if run.running and not run.stopping:
                run.stopping = True
                self._jobs.put(("stop", name))

    # ------------------------------------------------------------------------- usage, views, events

    def _flush_usage(self) -> None:
        rows = []
        for name, run in self.runs.items():
            if run.usage_dirty:
                rows.append((name, run.calls, run.errors, run.last_used))
                run.calls = run.errors = 0
                run.usage_dirty = False
        if rows:
            self.db.many("INSERT INTO usage (model_id, calls, errors, last_used) VALUES (?, ?, ?, ?) ON CONFLICT(model_id) "
                         "DO UPDATE SET calls = calls + excluded.calls, errors = errors + excluded.errors, "
                         "last_used = MAX(COALESCE(last_used, 0), excluded.last_used)", rows)

    def model_view(self, name: str) -> dict[str, Any]:
        """state, detail, leased_by, in_flight, build for one model (under the lock)."""
        with self.lock:
            run = self.runs.get(name) or Run()
            queued = [r for r in self._live() if r.state == "queued" and name in r.models]
            if run.stopping:
                state = "stopping"
            elif run.starting:
                state = "starting"
            elif run.running:
                state = "running"
            elif queued:
                state = "queued"
            elif run.failed:
                state = "failed"
            else:
                state = "stopped"
            detail = None
            if state == "queued":
                detail = f"position {queued[0].position}: {queued[0].waiting_for}"
            elif state == "failed":
                detail = run.failed
            leased_by = sorted({r.holder for r in self._live() if r.kind == "lease" and r.state != "queued"
                                and name in r.models})
            return {"state": state, "state_detail": detail, "leased_by": leased_by, "in_flight": run.in_flight,
                    "build": run.build, "ready_at": run.ready_at, "last_used": run.last_used}

    def _publish(self, p: Platform) -> None:
        for s in p.services:
            v = self.model_view(s.name)
            b: Build | None = v["build"]
            snap = {"id": s.name, "state": v["state"], "state_detail": v["state_detail"],
                    "build_key": b.build_key if b else None, "leased_by": v["leased_by"]}
            if self._last_published.get("m:" + s.name) != snap:
                self._last_published["m:" + s.name] = snap
                self.events.publish("model", snap)
        q = [{"id": r.id, "holder": r.holder, "position": r.position, "waiting_for": r.waiting_for, "state": r.state}
             for r in self._live()]
        if self._last_published.get("queue") != q:
            self._last_published["queue"] = q
            self.events.publish("queue", {"queue": q})
        for r in self.requests.values():
            snap = r.public()
            if self._last_published.get("r:" + r.id) != snap:
                self._last_published["r:" + r.id] = snap
                self.events.publish("lease", snap)
        # forget ended requests after they were published once; the DB keeps them
        for rid in [rid for rid, r in self.requests.items() if r.state not in LIVE]:
            self.requests.pop(rid)
            self._last_published.pop("r:" + rid, None)
