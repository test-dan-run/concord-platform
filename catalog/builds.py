"""Current build of each model: image id + config hash + weights hash -> build key (tools/identity.py).

Weights folders are hashed by content once, in a background thread, and cached by a quick
fingerprint (path, size, mtime of each file); until that hash exists the build key is None.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools import identity
from tools.schema import Manifest, Platform

from .db import DB, now_iso
from .runtime import ContainerInfo, Runtime

log = logging.getLogger("catalog.builds")
QUICK_TTL_S = 60.0


@dataclass
class Build:
    build_key: str | None
    image_id: str | None
    config_hash: str
    weights_hash: str | None
    pending: str | None  # why there is no build key yet


class WeightsHasher:
    def __init__(self, db: DB):
        self.db = db
        self._quick: dict[str, tuple[float, str]] = {}
        self._queued: set[str] = set()
        self._q: queue.Queue[Path] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, daemon=True, name="weights-hasher")
        self._thread.start()

    def quick(self, path: Path) -> str:
        key = str(path)
        with self._lock:
            hit = self._quick.get(key)
            if hit and time.monotonic() - hit[0] < QUICK_TTL_S:
                return hit[1]
        fp = identity.quick_fingerprint(path)
        with self._lock:
            self._quick[key] = (time.monotonic(), fp)
        return fp

    def get(self, path: Path) -> str | None:
        """The content hash if known for the folder as it is now; otherwise queue hashing and return None."""
        quick = self.quick(path)
        row = self.db.one("SELECT quick, content FROM weights_hash WHERE path = ?", (str(path),))
        if row and row["quick"] == quick:
            return row["content"]
        with self._lock:
            if str(path) not in self._queued:
                self._queued.add(str(path))
                self._q.put(path)
        return None

    def hashing(self, path: Path) -> bool:
        with self._lock:
            return str(path) in self._queued

    def _work(self) -> None:
        while True:
            path = self._q.get()
            try:
                t0 = time.monotonic()
                quick = identity.quick_fingerprint(path)
                content = identity.content_hash(path)
                if identity.quick_fingerprint(path) == quick:  # unchanged while hashing
                    self.db.execute("INSERT INTO weights_hash (path, quick, content, computed_at) VALUES (?, ?, ?, ?) "
                                    "ON CONFLICT(path) DO UPDATE SET quick = excluded.quick, content = excluded.content, "
                                    "computed_at = excluded.computed_at", (str(path), quick, content, now_iso()))
                    with self._lock:
                        self._quick[str(path)] = (time.monotonic(), quick)
                    log.info("hashed weights %s in %.1fs", path, time.monotonic() - t0)
            except OSError as exc:
                log.warning("cannot hash weights %s: %s", path, exc)
            except Exception:  # noqa: BLE001 - e.g. a locked database: keep hashing the other folders
                log.exception("cannot hash weights %s", path)
            finally:
                with self._lock:
                    self._queued.discard(str(path))


class Builds:
    def __init__(self, db: DB, runtime: Runtime):
        self.db = db
        self.runtime = runtime
        self.hasher = WeightsHasher(db)

    def weights_hash(self, p: Platform, s: Manifest) -> tuple[str | None, str | None]:
        """(hash, why it is missing)."""
        src = identity.weights_source(p, s)
        if src.kind == "none":
            return "none", None
        assert src.path is not None
        if src.kind == "yaml":
            return identity.weights_yaml_hash(src.path), None
        if not src.path.is_dir():
            return None, f"weights folder {src.path} not found"
        h = self.hasher.get(src.path)
        return (h, None) if h else (None, "hashing weights")

    def current(self, p: Platform, s: Manifest, container: ContainerInfo | None) -> Build:
        running = container is not None and container.state == "running"
        image_id = container.image_id if running and container.image_id else self.runtime.image_id(p.image(s))
        config = container.config_hash if running and container.config_hash else identity.config_hash(p, s)
        weights, why = self.weights_hash(p, s)
        if image_id is None:
            return Build(None, None, config, weights, f"image {p.image(s)} is not present locally")
        if weights is None:
            return Build(None, image_id, config, None, why)
        key = identity.build_key(image_id, config, weights)
        self.db.execute("INSERT OR IGNORE INTO build (build_key, model_id, image_id, config_hash, weights_hash, first_seen) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (key, s.name, image_id, config, weights, now_iso()))
        return Build(key, image_id, config, weights, None)

    def prewarm(self, p: Platform) -> None:
        """Queue content hashes of every weights folder, so build keys are ready before anyone asks."""
        for s in p.services:
            src = identity.weights_source(p, s)
            if src.kind == "dir" and src.path and src.path.is_dir():
                self.hasher.get(src.path)
