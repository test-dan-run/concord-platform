"""Export: pack platform models with their exact builds into a bundle, or push their images to a registry.

One job at a time, in a worker thread, with progress on the event stream (`export` events). The bundle
is a folder under Settings.export_dir:

    catalog-export-<date>/
      manifest.json        format version, models, build keys, image refs and ids, weights hashes, sizes
      images.tar           every selected image, each shared layer once (bundle destination only)
      models/<id>/
        service.yaml
        registration.json  display name, tags, derived from
        weights/           when weights are included
        verification.json  when verification records are included

The download route streams the folder as one .tar while it reads, so it is never stored twice. Hosted
endpoints are not exported: they have no image or weights. `python -m catalog import <bundle>` loads a
bundle on the target (catalog/importer.py).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools import identity
from tools.schema import Manifest, Platform

from .db import DB, dumps, now_iso
from .events import Events
from .registry import Registry
from .runtime import Cancelled, ImageError, Runtime
from .scheduler import Scheduler, SchedulerError
from .settings import Settings

log = logging.getLogger("catalog.export")
FORMAT = 1
DISK_MARGIN = 1.10  # free space needed: the estimate plus 10%
PROGRESS_S = 2.0
COPY_CHUNK = 8 << 20
LIVE = ("queued", "running")


@dataclass
class Weights:
    path: Path | None  # the local folder to copy, when there is one
    size: int
    files: int
    note: str | None  # why weights are left out, e.g. "downloaded on first start"


class Exporter:
    def __init__(self, db: DB, settings: Settings, scheduler: Scheduler, registry: Registry, runtime: Runtime,
                 events: Events):
        self.db = db
        self.settings = settings
        self.scheduler = scheduler
        self.registry = registry
        self.runtime = runtime
        self.events = events
        self._sizes: dict[str, tuple[int | None, Weights]] = {}  # build key -> (image bytes, weights)
        self._lock = threading.Lock()
        # held from the 409 check until the job's thread runs; not _lock, which models() takes through _sized
        self._start_lock = threading.Lock()
        self._job: str | None = None  # the job the thread runs
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._last_publish = 0.0

    # ------------------------------------------------------------------------- startup

    def recover(self) -> None:
        """Jobs left running by a restart failed: say so and remove their partial bundles."""
        for row in self.db.all("SELECT * FROM export_job WHERE state IN ('queued', 'running')"):
            self._remove(row.get("path"))
            self.db.execute("UPDATE export_job SET state = 'failed', error = ?, failed_phase = COALESCE(phase, 'starting'), "
                            "finished_at = ? WHERE id = ?", ("interrupted by a catalog restart", now_iso(), row["id"]))

    # ------------------------------------------------------------------------- what can be exported

    def _weights(self, p: Platform, s: Manifest) -> Weights:
        src = identity.weights_source(p, s)
        if src.kind == "none":
            return Weights(None, 0, 0, None)
        folder = src.path if src.kind == "dir" else p.weights_dir(s)
        if src.kind == "yaml":  # weights.yaml lists files fetched into the manifest's weights path
            import yaml

            listed = [str(f.get("path")) for f in (yaml.safe_load(src.path.read_text()) or {}).get("files") or []]
            if not folder or not folder.is_dir() or not all((folder / f).is_file() for f in listed):
                return Weights(None, 0, 0, "downloaded on first start")
        if not folder or not folder.is_dir():
            return Weights(None, 0, 0, f"weights folder {folder} not found")
        files = identity.weight_files(folder)
        return Weights(folder, sum(f.stat().st_size for _, f in files), len(files), None)

    def _sized(self, p: Platform, s: Manifest, build_key: str | None) -> tuple[int | None, Weights]:
        """Image and weights sizes, cached per build key: inspecting images and walking folders is slow."""
        with self._lock:
            hit = self._sizes.get(build_key or "") if build_key else None
        if hit:
            return hit
        out = (self.runtime.image_size(p.image(s)), self._weights(p, s))
        if build_key:
            with self._lock:
                self._sizes[build_key] = out
        return out

    def models(self) -> list[dict[str, Any]]:
        """GET /api/export/models: every platform model with sizes and whether it can be exported."""
        p = self.scheduler.platform()
        regs = {r["model_id"] for r in self.db.all("SELECT model_id FROM registration")}
        out = []
        for s in p.services:
            rec = self.registry.record(s.name, p)
            image_size, w = self._sized(p, s, rec["build_key"])
            reason = None
            if image_size is None:
                reason = f"image {p.image(s)} is not present locally"
            elif not rec["build_key"]:
                reason = f"no build key yet: {rec['build_pending']}"
            out.append({"id": s.name, "display_name": rec["display_name"], "registered": s.name in regs,
                        "contract": s.contract, "api": s.api, "build_key": rec["build_key"], "image": p.image(s),
                        "image_size": image_size, "weights_size": w.size, "weights_files": w.files,
                        "weights_note": w.note, "state": rec["state"],
                        "verification": (rec["capabilities"] or {}).get("status"),
                        "exportable": reason is None, "reason": reason})
        return out

    # ------------------------------------------------------------------------- jobs

    def _row(self, job_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM export_job WHERE id = ?", (job_id,))
        if row is None:
            raise SchedulerError(404, f"unknown export {job_id!r}")
        return row

    def public(self, row: dict[str, Any]) -> dict[str, Any]:
        req = json.loads(row["request"])
        path = Path(row["path"]) if row["path"] else None
        return {"id": row["id"], "state": row["state"], "phase": row["phase"], "phase_index": row["phase_index"],
                "phase_count": row["phase_count"], "bytes_done": row["bytes_done"],
                "bytes_expected": row["bytes_expected"], "bytes_written": row["bytes_written"], "error": row["error"],
                "failed_phase": row["failed_phase"], "request": req,
                "name": path.name if path else None,
                "available": row["state"] == "done" and path is not None and path.is_dir(),
                "created_at": row["created_at"], "finished_at": row["finished_at"]}

    def jobs(self) -> list[dict[str, Any]]:
        return [self.public(r) for r in self.db.all("SELECT * FROM export_job ORDER BY created_at DESC, rowid DESC")]

    def get(self, job_id: str) -> dict[str, Any]:
        return self.public(self._row(job_id))

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._start_lock:  # two starts at once: the second one sees the first one's job and gets 409
            return self._start(body)

    def _start(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.db.one("SELECT 1 FROM export_job WHERE state IN ('queued', 'running')"):
            raise SchedulerError(409, "an export is already running; wait for it or cancel it")
        p = self.scheduler.platform()
        models = list(dict.fromkeys(body.get("models") or []))
        if not models:
            raise SchedulerError(422, "select at least one model")
        hosted = [m for m in models if m in self.registry.hosted]
        if hosted:
            raise SchedulerError(422, f"hosted endpoints have no image or weights to export: {', '.join(hosted)}")
        try:
            names = [p.get(m).name for m in models]
        except KeyError as exc:
            raise SchedulerError(404, exc.args[0]) from None
        dest = body.get("destination") or "bundle"
        if dest not in ("bundle", "registry"):
            raise SchedulerError(422, "destination must be bundle or registry")
        registry = (body.get("registry") or "").strip().rstrip("/")
        include = {"images": bool(body.get("images", True)), "weights": bool(body.get("weights", True)),
                   "verification": bool(body.get("verification", True))}
        if dest == "registry" and include["images"] and not registry:
            raise SchedulerError(422, "enter the target registry to push to")
        bad = [m for m in self.models() if m["id"] in names and not m["exportable"]]
        if bad:
            raise SchedulerError(422, "; ".join(f"{m['id']}: {m['reason']}" for m in bad))
        req = {"models": names, "destination": dest, "registry": registry or None, **include}
        job_id = f"x-{uuid.uuid4().hex[:10]}"
        path = self._bundle_path()
        self.db.execute("INSERT INTO export_job (id, request, state, phase, path, created_at) VALUES (?, ?, 'queued', ?, ?, ?)",
                        (job_id, dumps(req), "queued", str(path), now_iso()))
        self._job, self._cancel = job_id, threading.Event()
        self._thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name=f"export-{job_id}")
        self._thread.start()
        return self.get(job_id)

    def _bundle_path(self) -> Path:
        base = f"catalog-export-{datetime.now(UTC).strftime('%Y-%m-%d')}"
        root = self.settings.export_dir
        path, n = root / base, 2
        while path.exists() or self.db.one("SELECT 1 FROM export_job WHERE path = ?", (str(path),)):
            path, n = root / f"{base}-{n}", n + 1
        return path

    def cancel_or_delete(self, job_id: str) -> None:
        """A running job is cancelled; any other job (a live one included, when no thread runs it) is deleted."""
        with self._start_lock:
            row = self._row(job_id)
            running = row["state"] in LIVE and job_id == self._job
        if running:
            self._cancel.set()
            return
        self._remove(row["path"])
        self.db.execute("DELETE FROM export_job WHERE id = ?", (job_id,))
        self._publish(job_id, force=True)

    def wait(self, timeout: float = 10.0) -> None:
        """Tests: wait for the running job's thread."""
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------------- the worker

    def _set(self, job_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE export_job SET {sets} WHERE id = ?", (*fields.values(), job_id))

    def _publish(self, job_id: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < PROGRESS_S:
            return
        self._last_publish = now
        row = self.db.one("SELECT * FROM export_job WHERE id = ?", (job_id,))
        self.events.publish("export", self.public(row) if row else {"id": job_id, "state": "deleted"})

    def _remove(self, path: str | None) -> None:
        if path and Path(path).exists() and Path(self.settings.export_dir) in Path(path).resolve().parents:
            shutil.rmtree(path, ignore_errors=True)

    def _run(self, job_id: str) -> None:
        row = self._row(job_id)
        req = json.loads(row["request"])
        out = Path(row["path"])
        bundle = req["destination"] == "bundle"
        phases = ["checking disk space"]
        if req["images"]:
            phases.append("saving images" if bundle else "pushing images")
        if req["weights"]:
            phases.append("copying weights")
        phases.append("writing records")
        phase = phases[0]
        p = self.scheduler.platform()
        services = [p.get(m) for m in req["models"]]
        sized = {s.name: self._sized(p, s, self.registry.record(s.name, p)["build_key"]) for s in services}
        images = list(dict.fromkeys(p.image(s) for s in services))
        # "up to": images that share layers make images.tar smaller than the sum of their sizes
        image_sum = sum(self.runtime.image_size(ref) or 0 for ref in images)
        weights_sum = sum(sized[s.name][1].size for s in services) if req["weights"] else 0
        expected = (image_sum if bundle and req["images"] else 0) + weights_sum
        done = 0

        def enter(name: str, detail: str | None = None) -> None:
            nonlocal phase
            phase = name
            self._set(job_id, state="running", phase=detail or name, phase_index=phases.index(name) + 1,
                      phase_count=len(phases), bytes_done=done, bytes_expected=expected)
            self._publish(job_id, force=True)

        def progress(n: int) -> None:
            self._set(job_id, bytes_done=n)
            self._publish(job_id)

        created = False  # a folder this job did not make is never removed

        def remove() -> None:
            if created:
                self._remove(str(out))

        try:
            enter("checking disk space")
            out.mkdir(parents=True, exist_ok=False)
            created = True
            free = shutil.disk_usage(out).free
            if free < expected * DISK_MARGIN:
                raise ExportError(f"not enough free space in {self.settings.export_dir}: {_gb(free)} free, "
                                  f"{_gb(expected * DISK_MARGIN)} needed (the estimate plus 10%)")
            records: dict[str, dict[str, Any]] = {s.name: {} for s in services}
            if req["images"] and bundle:
                enter("saving images", "saving images (one archive, shared layers once)")
                tar = out / "images.tar"
                self.runtime.save_images(images, tar, self._cancel,
                                         lambda: progress(done + (tar.stat().st_size if tar.exists() else 0)))
                done += tar.stat().st_size
                progress(done)
            elif req["images"]:
                enter("pushing images")
                for k, ref in enumerate(images, 1):
                    self._set(job_id, phase=f"pushing image {k} of {len(images)}")
                    self._publish(job_id, force=True)
                    target = f"{req['registry']}/{ref.rsplit('/', 1)[-1]}"
                    digest = self.runtime.push_image(ref, target, self._cancel)
                    for s in services:
                        if p.image(s) == ref:
                            records[s.name]["pushed"] = {"ref": target, "digest": digest}
            if req["weights"]:
                enter("copying weights")
                for s in services:
                    w = sized[s.name][1]
                    if w.path is None:
                        continue
                    self._set(job_id, phase=f"copying weights for {s.name}")
                    self._publish(job_id, force=True)
                    try:
                        done = self._copy_weights(w.path, out / "models" / s.name / "weights", done, progress)
                    except OSError as exc:
                        raise ExportError(f"{s.name}: {exc.strerror or exc}") from None
            enter("writing records")
            self._write_records(p, out, services, sized, records, req)
            written = _tree_size(out)
            self._set(job_id, state="done", phase="done", bytes_done=written, bytes_written=written, finished_at=now_iso())
            log.info("export %s done: %s (%s)", job_id, out, _gb(written))
        except Cancelled:
            remove()
            self._set(job_id, state="cancelled", error="cancelled", failed_phase=phase, finished_at=now_iso())
        except (ExportError, ImageError, OSError) as exc:
            log.warning("export %s failed while %s: %s", job_id, phase, exc)
            remove()
            msg = str(exc) if not isinstance(exc, OSError) else f"{exc.strerror or exc}"
            self._set(job_id, state="failed", error=msg, failed_phase=phase, finished_at=now_iso())
        except Exception as exc:  # noqa: BLE001 - a bug must still end the job
            log.exception("export %s crashed", job_id)
            remove()
            self._set(job_id, state="failed", error=f"export crashed: {exc}", failed_phase=phase, finished_at=now_iso())
        self._publish(job_id, force=True)

    def _copy_weights(self, src: Path, dst: Path, done: int, progress) -> int:
        for rel, f in identity.weight_files(src):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "rb") as fin, open(target, "wb") as fout:
                while block := fin.read(COPY_CHUNK):
                    if self._cancel.is_set():
                        raise Cancelled("copying weights")
                    fout.write(block)
                    done += len(block)
                    progress(done)
            shutil.copystat(f, target)
        return done

    def _write_records(self, p: Platform, out: Path, services: list[Manifest], sized: dict[str, tuple[int | None, Weights]],
                       extra: dict[str, dict[str, Any]], req: dict[str, Any]) -> None:
        entries = []
        for s in services:
            d = out / "models" / s.name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.files[s.name], d / "service.yaml")
            rec = self.registry.record(s.name, p)
            reg = self.db.one("SELECT * FROM registration WHERE model_id = ?", (s.name,))
            (d / "registration.json").write_text(json.dumps(
                {"id": s.name, "registered": reg is not None, "display_name": (reg or {}).get("display_name"),
                 "tags": json.loads(reg["tags"]) if reg else [], "parent_id": (reg or {}).get("parent_id")}, indent=2))
            caps = rec["capabilities"]
            if req["verification"] and caps:
                (d / "verification.json").write_text(json.dumps(caps, indent=2))
            build = self.db.one("SELECT * FROM build WHERE build_key = ?", (rec["build_key"],)) if rec["build_key"] else None
            image_size, w = sized[s.name]
            entries.append({
                "id": s.name, "build_key": rec["build_key"], "image": p.image(s),
                "image_id": (build or {}).get("image_id"), "image_size": image_size,
                "config_hash": (build or {}).get("config_hash"), "weights_hash": (build or {}).get("weights_hash"),
                "weights": {"included": bool(req["weights"] and w.path), "size": w.size if req["weights"] and w.path else 0,
                            "files": w.files if req["weights"] and w.path else 0, "note": w.note,
                            "source": identity.weights_source(p, s).kind},
                "verification": caps["status"] if caps else None, **extra.get(s.name, {})})
        manifest = {"format": FORMAT, "created_at": now_iso(), "source": self.settings.public_url,
                    "destination": req["destination"], "registry": req.get("registry"),
                    "images": "images.tar" if req["images"] and req["destination"] == "bundle" else None,
                    "include": {k: req[k] for k in ("images", "weights", "verification")}, "models": entries}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------------------- download

    def download(self, job_id: str) -> tuple[str, Iterator[bytes]]:
        """(file name, chunks): the bundle folder as one .tar, written into a pipe while it is read."""
        row = self._row(job_id)
        path = Path(row["path"] or "")
        if row["state"] != "done" or not path.is_dir():
            raise SchedulerError(409 if row["state"] in LIVE else 410, f"export {job_id} has no bundle to download")
        r, w = os.pipe()

        def writer() -> None:
            with os.fdopen(w, "wb", buffering=1 << 20) as fh:
                try:
                    with tarfile.open(fileobj=fh, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                        tar.add(path, arcname=path.name)
                except (BrokenPipeError, OSError):  # the client went away
                    pass

        threading.Thread(target=writer, daemon=True, name=f"download-{job_id}").start()

        def chunks() -> Iterator[bytes]:
            with os.fdopen(r, "rb") as fh:
                while block := fh.read(1 << 20):
                    yield block

        return f"{path.name}.tar", chunks()


class ExportError(Exception):
    pass


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _tree_size(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
