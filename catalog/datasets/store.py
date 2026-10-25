"""The audio store: original audio bytes keyed by the hex sha256 of those bytes (DATASET_CATALOG_PLAN.md §4.1).

    audio/<sha256[0:2]>/<sha256>   no extension: two uploads of the same bytes can be .wav and .WAV
    tmp/<sha256>                   an upload in progress

Only the catalog service writes here. An upload is a series of PUTs, each starting at the size the store already
has, so an upload resumes after a dropped connection. Commit checks the sha256, sets the mode to 0644 and moves the
file into place, so an app reading the mount never sees a partial file. The extension, size and what ffprobe finds
(duration, sample rate, channels) are kept in datasets.db.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import IO, Any

from ..db import DB, now_iso
from . import DatasetError

log = logging.getLogger("catalog.datasets")
HEX64 = re.compile(r"[0-9a-f]{64}")  # with fullmatch: a "$" would also match before a final newline
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma", ".webm"}
MEDIA_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
               ".opus": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".wma": "audio/x-ms-wma",
               ".webm": "audio/webm", ".zip": "application/zip"}
COPY_CHUNK = 1 << 20
PROBE_TIMEOUT_S = 60


def check_sha(sha: str) -> str:
    if not isinstance(sha, str) or not HEX64.fullmatch(sha):
        raise DatasetError(422, f"{sha!r} is not a sha256: 64 lowercase hex characters", "invalid_request")
    return sha


def norm_ext(ext: str | None) -> str:
    """'.wav' from 'wav', '.WAV' or '.wav'; '' when there is none."""
    ext = (ext or "").strip().lower().lstrip(".")
    if ext and not re.fullmatch(r"[a-z0-9]{1,10}", ext):
        raise DatasetError(422, f"bad extension {ext!r}", "invalid_request")
    return f".{ext}" if ext else ""


def rel_ok(rel: str) -> bool:
    """A relative POSIX path without '..' (paths inside packages and zips)."""
    p = PurePosixPath(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts and "\\" not in rel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(COPY_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in ("sha256", "size", "ext", "duration", "sample_rate", "channels", "state", "created_at")}


class AudioStore:
    def __init__(self, root: Path, db: DB):
        self.root = Path(root)
        self.db = db
        # Publishing, deleting and clean-up take this lock, so clean-up never removes audio a publish is adding.
        self.lock = threading.RLock()
        self._busy: set[str] = set()  # sha256s with a PUT or a commit running
        self._busy_lock = threading.Lock()
        self.ffprobe = shutil.which("ffprobe")
        for d in ("audio", "tmp", "versions"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------- paths and records

    def path(self, sha: str) -> Path:
        return self.root / "audio" / sha[:2] / sha

    def tmp_path(self, sha: str) -> Path:
        return self.root / "tmp" / sha

    def scratch(self) -> Path:
        """A new empty folder under tmp/, on the store's filesystem so its files can be moved into place."""
        d = self.root / "tmp" / f"work-{uuid.uuid4().hex}"
        d.mkdir()
        return d

    def record(self, sha: str) -> dict[str, Any] | None:
        return _record(self.db.one("SELECT * FROM audio WHERE sha256 = ?", (sha,)))

    def ready(self, shas: Iterable[str]) -> dict[str, dict[str, Any]]:
        """The records of those sha256s the store has whole."""
        out: dict[str, dict[str, Any]] = {}
        shas = list(dict.fromkeys(shas))
        for i in range(0, len(shas), 500):
            part = shas[i:i + 500]
            rows = self.db.all(f"SELECT * FROM audio WHERE state = 'ready' AND sha256 IN ({','.join('?' * len(part))})",
                               tuple(part))
            out.update({r["sha256"]: _record(r) for r in rows})
        return out

    def missing(self, shas: list[str]) -> list[str]:
        """The sha256s the store doesn't have, or has only partly, in the order asked."""
        for s in shas:
            check_sha(s)
        have = self.ready(shas)
        return [s for s in dict.fromkeys(shas) if s not in have]

    def require(self, sha: str) -> dict[str, Any]:
        rec = self.record(check_sha(sha))
        if rec is None or rec["state"] != "ready":
            raise DatasetError(404, f"the store has no audio {sha}", "not_found")
        return rec

    def media_type(self, rec: dict[str, Any]) -> str:
        return MEDIA_TYPES.get(rec["ext"], "application/octet-stream")

    # ------------------------------------------------------------------------- uploads

    @contextlib.contextmanager
    def _claim(self, sha: str) -> Iterator[None]:
        with self._busy_lock:
            if sha in self._busy:
                raise DatasetError(409, f"another upload of {sha} is running", "upload_busy", size=self._size(sha))
            self._busy.add(sha)
        try:
            yield
        finally:
            with self._busy_lock:
                self._busy.discard(sha)

    def busy(self, sha: str) -> bool:
        with self._busy_lock:
            return sha in self._busy

    def _size(self, sha: str) -> int:
        rec = self.record(sha)
        if rec and rec["state"] == "ready":
            return rec["size"]
        t = self.tmp_path(sha)
        return t.stat().st_size if t.is_file() else 0

    @contextlib.contextmanager
    def upload(self, sha: str, ext: str | None, offset: int) -> Iterator[IO[bytes] | None]:
        """Append to an upload, starting at `offset`. Yields the open file, or None when the store already has the
        audio whole. 409 (with the current size) when `offset` isn't the size the store has, so the caller resumes
        from there."""
        check_sha(sha)
        ext = norm_ext(ext)
        with self._claim(sha):
            rec = self.record(sha)
            if rec and rec["state"] == "ready":
                yield None
                return
            size = self._size(sha)
            if offset != size:
                raise DatasetError(409, f"the store has {size} bytes of {sha}, not {offset}", "offset_mismatch",
                                   size=size)
            if rec is None:
                self.db.execute("INSERT OR IGNORE INTO audio (sha256, size, ext, state, created_at) "
                                "VALUES (?, 0, ?, 'partial', ?)", (sha, ext, now_iso()))
            elif ext and not rec["ext"]:
                self.db.execute("UPDATE audio SET ext = ? WHERE sha256 = ?", (ext, sha))
            with open(self.tmp_path(sha), "ab") as fh:
                yield fh
            self.db.execute("UPDATE audio SET size = ? WHERE sha256 = ? AND state = 'partial'", (self._size(sha), sha))

    def upload_size(self, sha: str) -> int:
        return self._size(sha)

    def commit(self, sha: str) -> dict[str, Any]:
        """Check the uploaded bytes against `sha`, probe them and move them into place. 422 on a mismatch, and the
        partial file is removed so the upload starts again."""
        check_sha(sha)
        with self._claim(sha):
            rec = self.record(sha)
            if rec and rec["state"] == "ready":
                return rec
            tmp = self.tmp_path(sha)
            if rec is None or not tmp.is_file():
                raise DatasetError(404, f"no upload of {sha} to commit", "not_found")
            actual = sha256_file(tmp)
            if actual != sha:
                tmp.unlink(missing_ok=True)
                self.db.execute("DELETE FROM audio WHERE sha256 = ? AND state = 'partial'", (sha,))
                raise DatasetError(422, f"the uploaded bytes hash to {actual}, not {sha}; the upload was removed",
                                   "hash_mismatch", actual=actual)
            return self._place(tmp, sha, rec["ext"])

    def put_file(self, src: Path, ext: str | None, sha: str | None = None) -> dict[str, Any]:
        """Store a file already on the store's filesystem (under tmp/), moving it. With `sha`, 422 if it doesn't
        match. A file the store already has is removed and the stored record returned."""
        actual = sha256_file(src)
        if sha is not None and actual != sha:
            src.unlink(missing_ok=True)
            raise DatasetError(422, f"{src.name}: the bytes hash to {actual}, not {sha}", "hash_mismatch",
                               actual=actual)
        with self._claim(actual):
            rec = self.record(actual)
            if rec and rec["state"] == "ready":
                src.unlink(missing_ok=True)
                return rec
            if rec is None:
                self.db.execute("INSERT INTO audio (sha256, size, ext, state, created_at) VALUES (?, 0, ?, 'partial', ?)",
                                (actual, norm_ext(ext), now_iso()))
            self.tmp_path(actual).unlink(missing_ok=True)  # a stale partial upload of the same bytes
            return self._place(src, actual, (rec or {}).get("ext") or norm_ext(ext))

    def _place(self, src: Path, sha: str, ext: str) -> dict[str, Any]:
        probe = self.probe(src)
        os.chmod(src, 0o644)
        dest = self.path(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)
        self.db.execute("UPDATE audio SET size = ?, ext = ?, duration = ?, sample_rate = ?, channels = ?, "
                        "state = 'ready', created_at = ? WHERE sha256 = ?",
                        (dest.stat().st_size, ext, probe["duration"], probe["sample_rate"], probe["channels"],
                         now_iso(), sha))
        return self.record(sha)  # type: ignore[return-value]

    def probe(self, path: Path) -> dict[str, Any]:
        """Duration, sample rate and channels from ffprobe; all None when ffprobe is missing or can't read it."""
        none = {"duration": None, "sample_rate": None, "channels": None}
        if not self.ffprobe:
            log.warning("ffprobe is not installed: audio is stored without its duration and sample rate")
            return none
        cmd = [self.ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
               "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False)
            info = json.loads(out.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return none
        streams = info.get("streams") or []
        if out.returncode != 0 or not streams:
            return none
        s = streams[0]
        dur = (info.get("format") or {}).get("duration") or s.get("duration")
        try:
            return {"duration": round(float(dur), 3) if dur not in (None, "N/A") else None,
                    "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
                    "channels": int(s["channels"]) if s.get("channels") else None}
        except (TypeError, ValueError):
            return none

    def remove(self, sha: str) -> None:
        """Delete one audio file and its record (clean-up and unpacked zips only)."""
        self.path(sha).unlink(missing_ok=True)
        self.tmp_path(sha).unlink(missing_ok=True)
        self.db.execute("DELETE FROM audio WHERE sha256 = ?", (sha,))

    # ------------------------------------------------------------------------- zips of audio

    def unpack(self, sha: str) -> dict[str, Any]:
        """Store and probe each audio file in a committed zip. Files ffprobe can't read are listed under `skipped`
        and not stored. The zip itself is removed from the store afterwards, unless a version lists it or an app
        holds it."""
        self.require(sha)
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        work = self.scratch()
        try:
            try:
                zf = zipfile.ZipFile(self.path(sha))
            except zipfile.BadZipFile:
                raise DatasetError(422, f"{sha} is not a zip", "invalid_zip") from None
            with zf:
                for info in zf.infolist():
                    name = info.filename
                    p = PurePosixPath(name)
                    if info.is_dir() or "__MACOSX" in p.parts or p.name.startswith("."):
                        continue
                    if not rel_ok(name):
                        skipped.append({"path": name, "error": "unsafe path"})
                        continue
                    if p.suffix.lower() not in AUDIO_EXTS:
                        skipped.append({"path": name, "error": "not an audio file"})
                        continue
                    tmp = work / uuid.uuid4().hex
                    with zf.open(info) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, COPY_CHUNK)
                    if self.ffprobe and self.probe(tmp)["duration"] is None:
                        tmp.unlink()
                        skipped.append({"path": name, "error": "ffprobe can't read it"})
                        continue
                    files.append({"path": name, **self.put_file(tmp, p.suffix)})
        finally:
            shutil.rmtree(work, ignore_errors=True)
        with self.lock:
            used = self.db.one("SELECT 1 AS x FROM version_file WHERE sha256 = ? UNION SELECT 1 FROM hold "
                               "WHERE sha256 = ?", (sha, sha))
            if not used:
                self.remove(sha)
        return {"files": files, "skipped": skipped}

    # ------------------------------------------------------------------------- holds and stats

    def holds(self, app: str, add: list[str], remove: list[str]) -> int:
        """Place and release one app's holds; the number of sha256s the app holds afterwards."""
        app = app.strip()
        if not app:
            raise DatasetError(422, "app is required", "invalid_request")
        for s in [*add, *remove]:
            check_sha(s)
        now = now_iso()
        with self.lock, self.db.tx() as c:
            c.executemany("INSERT OR IGNORE INTO hold (app, sha256, created_at) VALUES (?, ?, ?)",
                          [(app, s, now) for s in add])
            c.executemany("DELETE FROM hold WHERE app = ? AND sha256 = ?", [(app, s) for s in remove])
        return self.db.one("SELECT COUNT(*) AS n FROM hold WHERE app = ?", (app,))["n"]  # type: ignore[index]

    def stats(self) -> dict[str, Any]:
        def group(where: str) -> dict[str, int]:
            r = self.db.one(f"SELECT COUNT(*) AS files, COALESCE(SUM(size), 0) AS bytes FROM audio a WHERE {where}")
            return {"files": r["files"], "bytes": r["bytes"]}  # type: ignore[index]

        listed = "EXISTS (SELECT 1 FROM version_file v WHERE v.sha256 = a.sha256)"
        held = "EXISTS (SELECT 1 FROM hold h WHERE h.sha256 = a.sha256)"
        ready = "a.state = 'ready'"
        holds = self.db.all("SELECT h.app, COUNT(*) AS files, COALESCE(SUM(a.size), 0) AS bytes FROM hold h "
                            "LEFT JOIN audio a ON a.sha256 = h.sha256 AND a.state = 'ready' GROUP BY h.app ORDER BY h.app")
        disk = shutil.disk_usage(self.root)
        return {"root": str(self.root),
                "audio": group(ready),
                "in_versions": group(f"{ready} AND {listed}"),
                "held_only": group(f"{ready} AND NOT {listed} AND {held}"),
                "neither": group(f"{ready} AND NOT {listed} AND NOT {held}"),
                "partial": group("a.state = 'partial'"),
                "holds": holds,
                "datasets": self.db.one("SELECT COUNT(*) AS n FROM dataset")["n"],  # type: ignore[index]
                "versions": self.db.one("SELECT COUNT(*) AS n FROM version")["n"],  # type: ignore[index]
                "disk": {"total": disk.total, "free": disk.free}}
