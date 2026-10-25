"""Verifier: one capability record per build, from a live run of probes.py.

Verification is an ordinary scheduler client: it takes a lease (holder `catalog:verify`), so it waits
its turn in the queue and may start the model. Registered models are verified on registration, on
demand, and whenever their build key changes (a build with no record yet).

A hosted endpoint takes no lease and has no build to wait for: it is probed through the provider with
the stored key (catalog/hosted.py), and re-verified every day, because a provider can change the model
behind the same name.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tools import render

from ..db import DB, dumps, now_iso
from ..hosted import redact, redact_all
from ..scheduler import Scheduler, SchedulerError
from . import probes

if TYPE_CHECKING:
    from ..hosted import Hosted, HostedStore

log = logging.getLogger("catalog.verifier")
HOLDER = "catalog:verify"
SCAN_S = 10.0
BUILD_WAIT_S = 30 * 60.0  # weights hashing of a large model after its first start


class Verifier:
    def __init__(self, db: DB, scheduler: Scheduler, auto: bool = True, hosted: HostedStore | None = None,
                 hosted_reverify_s: float = 24 * 3600.0):
        self.db = db
        self.scheduler = scheduler
        self.auto = auto
        self.hosted = hosted
        self.hosted_reverify_s = hosted_reverify_s
        self.pending: dict[str, str] = {}  # model -> pending | running
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="verifier")
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    def enqueue(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.pending:
                return
            self.pending[model_id] = "pending"
            self._order.append(model_id)
        self._wake.set()

    def status(self, model_id: str) -> str | None:
        with self._lock:
            return self.pending.get(model_id)

    def _loop(self) -> None:
        last_scan = 0.0
        while not self._stop.is_set():
            if self.auto and time.monotonic() - last_scan > SCAN_S:
                last_scan = time.monotonic()
                try:
                    self.scan()
                except Exception:  # noqa: BLE001
                    log.exception("verification scan failed")
            with self._lock:
                model = self._order.popleft() if self._order else None
                if model:
                    self.pending[model] = "running"
            if model is None:
                self._wake.wait(SCAN_S)
                self._wake.clear()
                continue
            try:
                self.run(model)
            except Exception as exc:  # noqa: BLE001
                log.exception("verification of %s failed", model)
                self._store_failure(model, f"verification crashed: {exc}")
            finally:
                with self._lock:
                    self.pending.pop(model, None)

    def scan(self) -> None:
        """Queue registered models whose current build has no record yet, and hosted models verified over a day ago."""
        hosted = self.hosted.ids() if self.hosted else set()
        for h in self.hosted.all() if self.hosted else []:
            row = self.db.one("SELECT verified_at, updated_at FROM capability WHERE build_key = ?", (h.build_key,))
            at = (row or {}).get("verified_at") or (row or {}).get("updated_at")
            if at is None or time.time() - _ts(at) > self.hosted_reverify_s:
                self.enqueue(h.id)
        for row in self.db.all("SELECT model_id FROM registration"):
            m = row["model_id"]
            if m in hosted:
                continue
            with self.scheduler.lock:
                run = self.scheduler.runs.get(m)
                bk = run.build.build_key if run and run.build else None
            if bk and not self.db.one("SELECT 1 FROM capability WHERE build_key = ?", (bk,)):
                self.enqueue(m)

    def _store(self, model: str, build_key: str, status: str, record: dict[str, Any] | None,
               report: dict[str, Any], rtf: float | None) -> None:
        self.db.execute("INSERT INTO capability (build_key, model_id, status, record, report, rtf, verified_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(build_key) DO UPDATE SET status = excluded.status, "
                        "record = excluded.record, report = excluded.report, rtf = excluded.rtf, "
                        "verified_at = excluded.verified_at, updated_at = excluded.updated_at",
                        (build_key, model, status, dumps(record) if record is not None else None, dumps(report), rtf,
                         now_iso(), now_iso()))

    def _store_failure(self, model: str, reason: str) -> None:
        h = self.hosted.get(model) if self.hosted and model in self.hosted else None
        if h is not None:
            bk = h.build_key
        else:
            with self.scheduler.lock:
                run = self.scheduler.runs.get(model)
                bk = run.build.build_key if run and run.build else None
        if bk:
            self._store(model, bk, "failed", None, {"ok": False, "lines": [{"name": "verification", "status": "fail",
                                                                           "reason": reason}], "sample": {}}, None)

    def run(self, model: str) -> None:
        if self.hosted is not None and model in self.hosted:
            h = self.hosted.get(model)
            if h is not None:
                self.run_hosted(h)
            return
        sched = self.scheduler
        log.info("verifying %s", model)
        try:
            req = sched.request(HOLDER, [model])
        except SchedulerError as exc:
            self._store_failure(model, str(exc))
            return
        try:
            while True:  # wait for the lease, heartbeating; the queue may take a while
                r = sched.heartbeat(req.id)
                if r.state == "active":
                    break
                if self._stop.wait(2.0):
                    return
            deadline = time.monotonic() + BUILD_WAIT_S
            while True:  # the build key needs the running container's image and the weights hash
                with sched.lock:
                    build = sched.runs[model].build
                if build and build.build_key:
                    break
                if time.monotonic() > deadline:
                    self._store_failure(model, f"no build key: {build.pending if build else 'unknown'}")
                    return
                sched.heartbeat(req.id)
                sched.wake()
                if self._stop.wait(3.0):
                    return
            p = sched.platform()
            meta = render.endpoint_entry(p, p.get(model))
            beat = threading.Event()

            def keep_alive() -> None:
                while not beat.wait(20):
                    try:
                        sched.heartbeat(req.id)
                    except SchedulerError:
                        return

            t = threading.Thread(target=keep_alive, daemon=True)
            t.start()
            try:
                rep = probes.verify(meta, sched.upstream_url(model))
            finally:
                beat.set()
            audio, _, _ = probes.clip()
            record = rep.record(len(audio) / 16000)
            self._store(model, build.build_key, "verified" if rep.ok else "failed", record, rep.report(), record["rtf"])
            log.info("%s %s (%s)", model, "verified" if rep.ok else "failed verification", build.build_key)
        except SchedulerError as exc:
            self._store_failure(model, f"lease lost: {exc}")
        finally:
            r = sched.get(req.id)
            if r.state == "failed":
                self._store_failure(model, f"could not start: {r.error}")
            sched.release(req.id)

    def run_hosted(self, h: Hosted) -> None:
        """No lease, no build wait: the probes run against the provider with the stored key."""
        assert self.hosted is not None
        log.info("verifying hosted %s (%s)", h.id, h.build_key)
        try:
            key = self.hosted.key(h)
        except SchedulerError as exc:
            self._store_failure(h.id, str(exc))
            return
        meta = {"name": h.model, "contract": h.contract, "api": h.api, "signature": _with_examples(h.signature), "hosted": True,
                # a hosted chat model is probed with audio, and verified as a text model if it takes none
                "tasks": ["audio"] if h.api == "chat.completions" else [], "audio_optional": True}
        try:
            rep = probes.verify(meta, "http://hosted", self.hosted.client(h, key))  # verify() closes the client
        except Exception as exc:  # noqa: BLE001 - logged and stored here, so the key can be redacted first
            log.error("verification of %s failed\n%s", h.id, redact(traceback.format_exc(), key).rstrip())
            self._store_failure(h.id, redact(f"verification crashed: {exc}", key))
            return
        audio, _, _ = probes.clip()
        # a provider may quote the key back in an error or an output: none of it is stored
        record = redact_all(rep.record(len(audio) / 16000), key)
        self._store(h.id, h.build_key, "verified" if rep.ok else "failed", record, redact_all(rep.report(), key),
                    record["rtf"])
        log.info("%s %s (%s)", h.id, "verified" if rep.ok else "failed verification", h.build_key)


def _with_examples(signature: dict[str, Any] | None) -> dict[str, Any] | None:
    """OIP metadata carries no example values; the platform's audio services declare SAMPLE_RATE = [16000]."""
    if not signature:
        return signature
    inputs = [dict(t, example=[16000]) if t.get("name") == "SAMPLE_RATE" and "example" not in t else t
              for t in signature.get("inputs", [])]
    return {**signature, "inputs": inputs}


def _ts(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
