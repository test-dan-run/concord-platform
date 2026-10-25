"""Client for the Dataset Catalog's API (platform/catalog/README.md, DATASET_CATALOG_PLAN.md §7): datasets, versions,
the audio store and holds.

    from aicore.datasets import DatasetCatalog, StaleParent
    dc = DatasetCatalog(app="concord-label")          # AICORE_CATALOG, else http://127.0.0.1:8700
    sha = dc.upload_audio("call.wav")                 # 16 MB chunks, resumes; skipped when the store has it
    dc.hold([sha])
    try:
        v = dc.publish(dataset_id, "out/version", parent=2)
    except StaleParent as exc:
        ...                                           # exc.latest: ask, then publish again with force=True
    dc.version_dir(dataset_id, v["n"])                # read versions and audio from the mounted store

The store is DATASET_STORE, else ${AI_DATA}/datasets, else ~/ai-data/datasets, the same default as the catalog
service. `FakeDatasetCatalog` has the same methods, in memory and in a folder, for app tests.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import uuid
import wave
import weakref
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Any

import httpx

from .catalog import DEFAULT_URL, CatalogError
from .dataset_format import (AUDIO_EXTS, FILES_HEADER, FORMAT, FORMAT_VERSION, HEX64, INSTALL_ID, check_folder, digest,
                             file_summary, parse, read_files_csv, rel_ok, sha256_file)
from .endpoints import catalog_url

CHUNK_BYTES = 16 << 20  # upload chunk
FILES_PAGE = 1000  # files per request when listing a version's files
SHA_BATCH = 1000  # sha256s per request for missing, hold and release
PUT_TRIES = 5  # attempts per chunk when the connection drops or another upload of the file is running
RETRY_S = 1.0

FilesArg = Mapping[str, str] | Iterable[Mapping[str, str]]


def store_path() -> Path:
    """The mounted store: DATASET_STORE, else ${AI_DATA}/datasets, else ~/ai-data/datasets."""
    data = Path(os.environ.get("AI_DATA") or Path.home() / "ai-data").expanduser()
    return Path(os.environ.get("DATASET_STORE") or data / "datasets").expanduser()


# ----------------------------------------------------------------------------- errors

class DatasetCatalogError(CatalogError):
    """An error from the Dataset Catalog. `type` is the error's type (e.g. "hash_mismatch") and `extra` its other
    fields (e.g. `size` on a 409 from an upload)."""

    def __init__(self, status: int, message: str, type_: str = "dataset_error", **extra: Any):
        super().__init__(status, message)
        self.type, self.extra = type_, extra


class NotFound(DatasetCatalogError):
    """404: no such dataset, version or audio."""


class StaleParent(DatasetCatalogError):
    """409: the parent isn't the latest version (or was deleted). Publish again with force=True to keep it."""

    @property
    def latest(self) -> dict[str, Any] | None:
        return self.extra.get("latest")


class HasReferences(DatasetCatalogError):
    """409 from publish_audio: the latest version has references, so new audio comes through an app."""

    @property
    def latest(self) -> dict[str, Any] | None:
        return self.extra.get("latest")


class MissingAudio(DatasetCatalogError):
    """422: some files are not in the store; upload them first."""

    @property
    def missing(self) -> list[str]:
        return list(self.extra.get("missing") or [])


class InvalidVersion(DatasetCatalogError):
    """422: the folder doesn't follow the version format (contracts/datasets/version.md)."""

    @property
    def problems(self) -> list[str]:
        return list(self.extra.get("problems") or [])


ERRORS = {"stale_parent": StaleParent, "has_references": HasReferences, "missing_audio": MissingAudio,
          "invalid_version": InvalidVersion}


def _error(status: int, message: str, type_: str = "dataset_error", **extra: Any) -> DatasetCatalogError:
    cls = NotFound if status == 404 else ERRORS.get(type_, DatasetCatalogError)
    return cls(status, message, type_, **extra)


def _files_arg(files: FilesArg) -> list[dict[str, str]]:
    """[{"path", "sha256"}] from that list or from {path: sha256}."""
    if isinstance(files, Mapping):
        return [{"path": p, "sha256": s} for p, s in files.items()]
    return [{"path": f["path"], "sha256": f["sha256"]} for f in files]


def _zip_folder(folder: Path, fh: IO[bytes]) -> None:
    """A version folder as a zip, at the zip's root. outputs/ is left out (the catalog drops it anyway)."""
    with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            rel = p.relative_to(folder).as_posix()
            if p.is_file() and rel.split("/")[0] != "outputs":
                zf.write(p, rel, zipfile.ZIP_STORED if rel.startswith("audio/") else zipfile.ZIP_DEFLATED)


class _Mount:
    """Paths in the mounted store (DATASET_CATALOG_PLAN.md §4): the apps read audio and versions from here."""

    store: Path

    def audio_path(self, sha256: str) -> Path:
        return self.store / "audio" / sha256[:2] / sha256

    def version_dir(self, dataset_id: str, n: int) -> Path:
        return self.store / "versions" / dataset_id / str(n)


# ----------------------------------------------------------------------------- the client

class DatasetCatalog(_Mount):
    """The Dataset Catalog's API on the catalog service. `app` names the calling app on everything it creates,
    publishes, holds and uses; `store` is where the store is mounted (store_path() by default)."""

    def __init__(self, url: str | None = None, *, app: str, store: str | Path | None = None,
                 http: httpx.Client | None = None, timeout: float = 120.0):
        if not app.strip():
            raise ValueError("app is required")
        self.url = (url or catalog_url() or DEFAULT_URL).rstrip("/")
        self.app = app.strip()
        self.store = Path(store).expanduser() if store else store_path()
        self._http = http or httpx.Client(base_url=self.url, timeout=timeout)

    # -------------------------------------------------------------------------- plumbing

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        r = self._http.request(method, "/api" + path, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                raise _error(r.status_code, r.text[:500]) from None
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict):
                extra = {k: v for k, v in err.items() if k not in ("message", "type", "code")}
                raise _error(r.status_code, str(err.get("message"))[:500], err.get("type") or "dataset_error", **extra)
            raise _error(r.status_code, str(err or (body.get("detail") if isinstance(body, dict) else body))[:500])
        return r

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = self._call(method, path, **kw)
        return r.json() if r.content else None

    def _created(self, r: httpx.Response) -> dict[str, Any]:
        return {**r.json(), "created": r.status_code == 201}

    def health(self) -> dict[str, Any]:
        return self._req("GET", "/health")

    def available(self) -> bool:
        try:
            self.health()
            return True
        except (CatalogError, httpx.HTTPError):
            return False

    # -------------------------------------------------------------------------- datasets

    def datasets(self, q: str | None = None, language: str | None = None, label_type: str | None = None,
                 app: str | None = None) -> list[dict[str, Any]]:
        """Datasets with their latest version, most recently changed first. `app` keeps those the app created,
        published to or uses."""
        params = {k: v for k, v in {"q": q, "language": language, "label_type": label_type, "app": app}.items() if v}
        return self._req("GET", "/datasets", params=params)

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        """The dataset with `latest` and `versions` (newest first)."""
        return self._req("GET", f"/datasets/{dataset_id}")

    def create_dataset(self, name: str, *, language: str | None = None, description: str | None = None,
                       tags: list[str] | None = None) -> dict[str, Any]:
        return self._req("POST", "/datasets", json={"name": name, "language": language, "description": description,
                                                    "tags": tags, "app": self.app})

    def update_dataset(self, dataset_id: str, *, name: str | None = None, description: str | None = None,
                       tags: list[str] | None = None) -> dict[str, Any]:
        body = {k: v for k, v in {"name": name, "description": description, "tags": tags}.items() if v is not None}
        return self._req("PATCH", f"/datasets/{dataset_id}", json=body)

    def delete_dataset(self, dataset_id: str) -> None:
        self._req("DELETE", f"/datasets/{dataset_id}")

    # -------------------------------------------------------------------------- versions

    def version(self, dataset_id: str, n: int) -> dict[str, Any]:
        return self._req("GET", f"/datasets/{dataset_id}/versions/{n}")

    def files(self, dataset_id: str, n: int) -> list[dict[str, Any]]:
        """Every file of a version: path, sha256, duration, sample_rate, segments, speakers and labels."""
        out: list[dict[str, Any]] = []
        while True:
            page = self._req("GET", f"/datasets/{dataset_id}/versions/{n}/files",
                             params={"offset": len(out), "limit": FILES_PAGE})
            out += page["files"]
            if not page["files"] or len(out) >= page["total"]:
                return out

    def publish(self, dataset_id: str, folder: str | Path, parent: int | None = None, *,
                force: bool = False) -> dict[str, Any]:
        """Publish a version folder (contracts/datasets/version.md) based on version `parent` (None for the first).
        Returns the version with `created`: False when it has the same content as the latest version, which is
        returned instead. Raises StaleParent, MissingAudio or InvalidVersion."""
        data = {"app": self.app, "force": "true" if force else "false"}
        if parent is not None:
            data["parent"] = str(parent)
        with tempfile.TemporaryFile() as fh:
            _zip_folder(Path(folder), fh)
            fh.seek(0)
            r = self._call("POST", f"/datasets/{dataset_id}/versions", data=data,
                           files={"file": ("version.zip", fh, "application/zip")})
        return self._created(r)

    def publish_audio(self, dataset_id: str, files: FilesArg, parent: int | None = None, *,
                      force: bool = False) -> dict[str, Any]:
        """A version with audio only: the latest version's files plus `files` ({path: sha256}, or a list of
        {"path", "sha256"}). Returns the version with `created`. Raises HasReferences when the latest version has
        references, and StaleParent or MissingAudio like publish."""
        body = {"files": _files_arg(files), "parent": parent, "app": self.app, "force": force}
        return self._created(self._call("POST", f"/datasets/{dataset_id}/versions/audio", json=body))

    def delete_version(self, dataset_id: str, n: int) -> None:
        self._req("DELETE", f"/datasets/{dataset_id}/versions/{n}")

    def add_use(self, dataset_id: str, n: int, ref: str) -> None:
        """Record that this app made its dataset `ref` from the version."""
        self._req("POST", f"/datasets/{dataset_id}/versions/{n}/uses", json={"app": self.app, "ref": str(ref)})

    def remove_use(self, dataset_id: str, n: int, ref: str) -> None:
        """Idempotent, and it works after the version was deleted."""
        self._req("DELETE", f"/datasets/{dataset_id}/versions/{n}/uses", params={"app": self.app, "ref": str(ref)})

    # -------------------------------------------------------------------------- audio store

    def missing(self, sha256s: Iterable[str]) -> list[str]:
        """The sha256s the store doesn't have whole, in the order given."""
        shas = list(dict.fromkeys(sha256s))
        out: list[str] = []
        for i in range(0, len(shas), SHA_BATCH):
            out += self._req("POST", "/store/missing", json={"sha256": shas[i:i + SHA_BATCH]})["missing"]
        return out

    def _holds(self, key: str, sha256s: Iterable[str]) -> int:
        shas, held = list(dict.fromkeys(sha256s)), 0
        for i in range(0, max(len(shas), 1), SHA_BATCH):
            held = self._req("POST", "/store/holds", json={"app": self.app, key: shas[i:i + SHA_BATCH]})["held"]
        return held

    def hold(self, sha256s: Iterable[str]) -> int:
        """Keep these files in the store for this app; the number of files the app holds afterwards."""
        return self._holds("add", sha256s)

    def release(self, sha256s: Iterable[str]) -> int:
        return self._holds("remove", sha256s)

    def upload_audio(self, path: str | Path, sha256: str | None = None) -> str:
        """Upload one original to the store unless it has it already; its sha256. Pass `sha256` when it is known, to
        skip hashing the file. Sent in 16 MB chunks; an upload resumes from the size the store has after a dropped
        connection or an earlier partial upload. Raises DatasetCatalogError (type hash_mismatch) when the bytes
        don't match `sha256`."""
        path = Path(path)
        sha = sha256 or sha256_file(path)
        if not self.missing([sha]):
            return sha
        ext, size = path.suffix, path.stat().st_size
        offset = self._put(sha, ext, 0, b"")  # learns the size of an earlier partial upload, sending nothing
        with open(path, "rb") as fh:
            while offset < size:
                fh.seek(offset)
                offset = self._put(sha, ext, offset, fh.read(CHUNK_BYTES))
        self._req("POST", f"/store/audio/{sha}/commit")
        return sha

    def _put(self, sha: str, ext: str, offset: int, data: bytes) -> int:
        """One PUT of an upload; the size the store has afterwards. A 409 carries the size the store has, which is
        where the upload goes on from."""
        for attempt in range(PUT_TRIES):
            last = attempt == PUT_TRIES - 1
            try:
                params = {"ext": ext, "offset": offset}
                return self._req("PUT", f"/store/audio/{sha}", params=params, content=data)["size"]
            except DatasetCatalogError as exc:
                if exc.status != 409 or "size" not in exc.extra:
                    raise
                if exc.type != "upload_busy":
                    return int(exc.extra["size"])
                if last:
                    raise
            except httpx.TransportError:  # dropped: the next PUT at this offset answers 409 if some bytes arrived
                if last:
                    raise
            time.sleep(RETRY_S)
        raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- the fake

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ext(ext: str | None) -> str:
    ext = (ext or "").strip().lower().lstrip(".")
    return f".{ext}" if ext else ""


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def _probe(path: Path) -> dict[str, Any]:
    """Duration, sample rate and channels of a WAV file; None for anything else (the catalog uses ffprobe)."""
    try:
        with wave.open(str(path), "rb") as w:
            return {"duration": round(w.getnframes() / w.getframerate(), 3), "sample_rate": w.getframerate(),
                    "channels": w.getnchannels()}
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        return {"duration": None, "sample_rate": None, "channels": None}


class FakeDatasetCatalog(_Mount):
    """DatasetCatalog for app tests: the same methods and rules, with records in memory and audio and versions in a
    folder laid out like the real store, so audio_path and version_dir work.

    `root` is that folder (a temporary folder, removed with the fake, when not given). `add_audio` puts bytes in the
    store directly. Audio is probed only when it is WAV; give `duration` and `sample_rate` to `add_audio` for other
    formats that go into publish_audio. `calls` records every call that changes something, as (name, args).
    """

    url = "fake://datasets"

    def __init__(self, root: str | Path | None = None, *, app: str = "test-app"):
        if root is None:
            root = tempfile.mkdtemp(prefix="fake-dataset-store-")
            weakref.finalize(self, shutil.rmtree, root, True)
        self.store = Path(root)
        self.app = app
        for d in ("audio", "versions", "tmp"):
            (self.store / d).mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, dict[str, Any]] = {}
        self._versions: dict[tuple[str, int], dict[str, Any]] = {}
        self._audio: dict[str, dict[str, Any]] = {}
        self._holds: dict[str, dict[str, str]] = {}  # app -> {sha256: created_at}
        self._uses: dict[tuple[str, int], dict[tuple[str, str], str]] = {}
        self.calls: list[tuple[str, tuple]] = []

    def _log(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def available(self) -> bool:
        return True

    # -------------------------------------------------------------------------- records

    def _ds(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self._datasets:
            raise _error(404, f"no dataset {dataset_id!r}", "not_found")
        return self._datasets[dataset_id]

    def _row(self, dataset_id: str, n: int) -> dict[str, Any]:
        self._ds(dataset_id)
        if (dataset_id, n) not in self._versions:
            raise _error(404, f"dataset {dataset_id} has no version {n}", "not_found")
        return self._versions[(dataset_id, n)]

    def _latest(self, dataset_id: str) -> dict[str, Any] | None:
        ns = [n for d, n in self._versions if d == dataset_id]
        return self._versions[(dataset_id, max(ns))] if ns else None

    def _version(self, row: dict[str, Any]) -> dict[str, Any]:
        did, n, parent = row["dataset_id"], row["n"], row["parent_n"]
        uses = sorted(((at, app, ref) for (app, ref), at in self._uses.get((did, n), {}).items()))
        return {**{k: v for k, v in row.items() if k != "files"},
                "parent_deleted": parent is not None and (did, parent) not in self._versions,
                "uses": [{"app": app, "ref": ref, "created_at": at} for at, app, ref in uses],
                **json.loads(json.dumps({k: row[k] for k in ("manifest", "counts", "label_types", "labels")}))}

    def _dataset(self, ds: dict[str, Any], *, versions: bool = False) -> dict[str, Any]:
        latest = self._latest(ds["id"])
        out = {**{k: v for k, v in ds.items() if k != "last_n"}, "tags": list(ds["tags"]),
               "latest": self._version(latest) if latest else None}
        if versions:
            out["versions"] = [self._version(self._versions[k])
                               for k in sorted((k for k in self._versions if k[0] == ds["id"]), reverse=True)]
        return out

    # -------------------------------------------------------------------------- datasets

    def datasets(self, q: str | None = None, language: str | None = None, label_type: str | None = None,
                 app: str | None = None) -> list[dict[str, Any]]:
        rows = sorted(self._datasets.values(), key=lambda d: d["name"])
        out = []
        for ds in sorted(rows, key=lambda d: d["updated_at"], reverse=True):
            if q and q.lower() not in " ".join([ds["name"], ds["description"] or "", json.dumps(ds["tags"])]).lower():
                continue
            if language and (ds["language"] or "").lower() != language.lower():
                continue
            if app and ds["created_by"] != app and app not in self._apps(ds["id"]):
                continue
            d = self._dataset(ds)
            if label_type and label_type not in ((d["latest"] or {}).get("label_types") or []):
                continue
            out.append(d)
        return out

    def _apps(self, dataset_id: str) -> set[str]:
        """Apps that published a version of the dataset or use one."""
        return {v["published_by"] for (d, _), v in self._versions.items() if d == dataset_id} | \
            {app for (d, _), uses in self._uses.items() if d == dataset_id for app, _ in uses}

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._dataset(self._ds(dataset_id), versions=True)

    def create_dataset(self, name: str, *, language: str | None = None, description: str | None = None,
                       tags: list[str] | None = None) -> dict[str, Any]:
        self._log("create_dataset", name)
        if not (name or "").strip():
            raise _error(422, "name is required", "invalid_request")
        did, now = uuid.uuid4().hex, _now()
        self._datasets[did] = {"id": did, "name": name.strip(), "language": language or None,
                               "description": description or None, "tags": sorted({str(t) for t in tags or []}),
                               "created_by": self.app, "created_at": now, "updated_at": now, "last_n": 0}
        return self.dataset(did)

    def update_dataset(self, dataset_id: str, *, name: str | None = None, description: str | None = None,
                       tags: list[str] | None = None) -> dict[str, Any]:
        self._log("update_dataset", dataset_id)
        ds = self._ds(dataset_id)
        sets: dict[str, Any] = {}
        if name is not None:
            if not name.strip():
                raise _error(422, "name can't be empty", "invalid_request")
            sets["name"] = name.strip()
        if description is not None:
            sets["description"] = description or None
        if tags is not None:
            sets["tags"] = sorted(set(tags))
        if sets:
            ds.update(sets, updated_at=_now())
        return self.dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        self._log("delete_dataset", dataset_id)
        self._ds(dataset_id)
        for key in [k for k in self._versions if k[0] == dataset_id]:
            del self._versions[key]
            self._uses.pop(key, None)
        del self._datasets[dataset_id]
        shutil.rmtree(self.store / "versions" / dataset_id, ignore_errors=True)

    # -------------------------------------------------------------------------- versions

    def version(self, dataset_id: str, n: int) -> dict[str, Any]:
        return self._version(self._row(dataset_id, n))

    def files(self, dataset_id: str, n: int) -> list[dict[str, Any]]:
        row = self._row(dataset_id, n)
        parsed = parse(self.version_dir(dataset_id, n))
        return [{**f, **file_summary(parsed, f["path"])} for f in row["files"]]

    def publish(self, dataset_id: str, folder: str | Path, parent: int | None = None, *,
                force: bool = False) -> dict[str, Any]:
        self._log("publish", dataset_id, parent, force)
        return self._publish(self._ds(dataset_id), Path(folder), parent, force)

    def _publish(self, ds: dict[str, Any], root: Path, parent: int | None, force: bool) -> dict[str, Any]:
        """The catalog's publish (catalog/datasets/versions.py) on a folder: the same check, rules and messages."""
        problems = check_folder(root, package=True, hash_audio=False)
        if problems:
            raise _error(422, "the version is invalid: " + "; ".join(problems[:10]), "invalid_version",
                         problems=problems)
        manifest, rows = json.loads((root / "manifest.json").read_text(encoding="utf-8")), read_files_csv(root)
        for r in rows:  # a folder with audio/ (a reference package) adds its audio to the store
            src = root / "audio" / r["path"]
            if src.is_file():
                actual = sha256_file(src)
                if actual != r["sha256"]:
                    raise _error(422, f"{src.name}: the bytes hash to {actual}, not {r['sha256']}", "hash_mismatch",
                                 actual=actual)
                self._put(src, PurePosixPath(r["path"]).suffix)
        d = digest(root)
        latest = self._latest(ds["id"])
        if latest and latest["digest"] == d:
            return {**self._version(latest), "created": False}
        self._check_parent(ds, latest, parent, force)
        missing = list(dict.fromkeys(r["sha256"] for r in rows if r["sha256"] not in self._audio))
        if missing:
            raise _error(422, f"{len(missing)} audio files are not in the store", "missing_audio", missing=missing)
        n, now = ds["last_n"] + 1, _now()
        manifest.update({"source": {"app": self.app, "install_id": INSTALL_ID, "dataset_uid": ds["id"],
                                    "dataset_name": ds["name"]},
                         "export_version": n, "created_at": now, "audio_included": "none"})
        labels = self._write_folder(self.version_dir(ds["id"], n), root, manifest)
        files = [{"path": r["path"], "sha256": r["sha256"], "duration": float(r["duration"]),
                  "sample_rate": int(r["sample_rate"])} for r in sorted(rows, key=lambda r: r["path"])]
        self._versions[(ds["id"], n)] = {
            "dataset_id": ds["id"], "n": n, "parent_n": parent, "digest": d,
            "format_version": manifest.get("format_version"), "label_types": manifest["label_types"],
            "labels": labels, "counts": manifest["counts"],
            "published_by": self.app, "published_at": now, "manifest": manifest, "files": files}
        ds.update(last_n=n, updated_at=now)
        return {**self.version(ds["id"], n), "created": True}

    def _check_parent(self, ds: dict[str, Any], latest: dict[str, Any] | None, parent: int | None,
                      force: bool) -> None:
        if parent is not None and not 1 <= parent <= ds["last_n"]:
            raise _error(422, f"dataset {ds['id']} never had a version {parent}", "invalid_request")
        latest_n = latest["n"] if latest else None
        if parent == latest_n or force:
            return
        if parent is not None and (ds["id"], parent) not in self._versions:
            msg = f"v{parent} was deleted"
        else:
            msg = f"the version is based on {f'v{parent}' if parent else 'no version'}"
        msg += f", but the latest version is {f'v{latest_n}' if latest_n else 'none'}; publish with force to keep it"
        raise _error(409, msg, "stale_parent", latest=self._version(latest) if latest else None)

    def _write_folder(self, dest: Path, src: Path, manifest: dict[str, Any]) -> list[str]:
        """Write a version folder from a checked one, through a temporary folder renamed into place, so a failure
        leaves nothing behind. The label names its references carry."""
        tmp = Path(tempfile.mkdtemp(prefix="version-", dir=self.store / "tmp"))
        try:
            (tmp / "refs").mkdir()
            (tmp / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            shutil.copyfile(src / "files.csv", tmp / "files.csv")
            if (src / "refs").is_dir():
                for p in (src / "refs").iterdir():
                    shutil.copyfile(p, tmp / "refs" / p.name)
            if not any((tmp / "refs").iterdir()):
                (tmp / "refs").rmdir()
            labels = parse(tmp)["labels"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return labels

    def publish_audio(self, dataset_id: str, files: FilesArg, parent: int | None = None, *,
                      force: bool = False) -> dict[str, Any]:
        self._log("publish_audio", dataset_id, parent, force)
        ds = self._ds(dataset_id)
        files = _files_arg(files)
        if not files:
            raise _error(422, "files must not be empty", "invalid_request")
        latest = self._latest(dataset_id)
        if latest and latest["label_types"]:
            raise _error(409, f"v{latest['n']} has references: add audio to this dataset through an app",
                         "has_references", latest=self._version(latest))
        by_path = {f["path"]: dict(f) for f in (latest or {}).get("files", [])}
        stems = {PurePosixPath(p).stem.lower(): p for p in by_path}
        for f in files:
            self._check_sha(f["sha256"])
        ready = self._audio
        missing, unreadable = [], []
        for f in files:
            path, sha = f["path"] or "", f["sha256"]
            if not rel_ok(path) or PurePosixPath(path).suffix.lower() not in AUDIO_EXTS:
                raise _error(422, f"bad path {path!r}", "invalid_request")
            if path in by_path:
                if by_path[path]["sha256"] != sha:
                    raise _error(422, f"{path} is already in the dataset with other audio", "invalid_request")
                continue
            stem = PurePosixPath(path).stem.lower()
            if stem in stems:
                raise _error(422, f"the stem of {path} repeats {stems[stem]}", "invalid_request")
            if sha not in ready:
                missing.append(sha)
                continue
            rec = ready[sha]
            if rec["duration"] is None or rec["sample_rate"] is None:
                unreadable.append(path)
                continue
            stems[stem] = path
            by_path[path] = {"path": path, "sha256": sha, "duration": rec["duration"],
                             "sample_rate": rec["sample_rate"]}
        if missing:
            raise _error(422, f"{len(missing)} audio files are not in the store", "missing_audio", missing=missing)
        if unreadable:
            raise _error(422, f"ffprobe can't read {len(unreadable)} files", "unreadable_audio", unreadable=unreadable)
        rows = sorted(by_path.values(), key=lambda r: r["path"])
        manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "name": ds["name"], "created_at": _now(),
                    **{k: ds[k] for k in ("language", "description") if ds[k]}, "tags": list(ds["tags"]),
                    "label_types": [],
                    "counts": {"files": len(rows), "segments": 0,
                               "hours": round(sum(float(r["duration"]) for r in rows) / 3600, 6)},
                    "audio_included": "none"}
        return self._publish_new(ds, manifest, rows, None, parent, force)

    def _publish_new(self, ds: dict[str, Any], manifest: dict[str, Any], rows: list[dict[str, Any]],
                     refs: Mapping[str, str] | None, parent: int | None, force: bool) -> dict[str, Any]:
        """Write a version folder from its parts under tmp/ and publish it."""
        work = Path(tempfile.mkdtemp(dir=self.store / "tmp"))
        try:
            (work / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            with open(work / "files.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, FILES_HEADER, lineterminator="\n", extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            for name, text in (refs or {}).items():
                (work / "refs").mkdir(exist_ok=True)
                (work / "refs" / name).write_text(text, encoding="utf-8")
            return self._publish(ds, work, parent, force)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def delete_version(self, dataset_id: str, n: int) -> None:
        self._log("delete_version", dataset_id, n)
        self._row(dataset_id, n)
        del self._versions[(dataset_id, n)]
        self._uses.pop((dataset_id, n), None)
        self._datasets[dataset_id]["updated_at"] = _now()
        shutil.rmtree(self.version_dir(dataset_id, n), ignore_errors=True)

    def add_use(self, dataset_id: str, n: int, ref: str) -> None:
        self._log("add_use", dataset_id, n, str(ref))
        self._row(dataset_id, n)
        app, ref = self.app.strip(), str(ref).strip()  # stripped, as the catalog stores them
        if not app or not ref:
            raise _error(422, "app and ref are required", "invalid_request")
        self._uses.setdefault((dataset_id, n), {}).setdefault((app, ref), _now())

    def remove_use(self, dataset_id: str, n: int, ref: str) -> None:
        self._log("remove_use", dataset_id, n, str(ref))
        self._uses.get((dataset_id, n), {}).pop((self.app.strip(), str(ref).strip()), None)

    # -------------------------------------------------------------------------- audio store

    @staticmethod
    def _check_sha(sha: str) -> str:
        if not isinstance(sha, str) or not HEX64.fullmatch(sha):
            raise _error(422, f"{sha!r} is not a sha256: 64 lowercase hex characters", "invalid_request")
        return sha

    def _put(self, src: Path, ext: str | None, sha: str | None = None, **probe: Any) -> str:
        actual = sha256_file(src)
        if sha is not None and actual != sha:
            raise _error(422, f"the uploaded bytes hash to {actual}, not {sha}; the upload was removed",
                         "hash_mismatch", actual=actual)
        if actual not in self._audio:
            dest = self.audio_path(actual)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            os.chmod(dest, 0o644)
            found = _probe(dest)
            found.update({k: v for k, v in probe.items() if v is not None})
            self._audio[actual] = {"sha256": actual, "size": dest.stat().st_size, "ext": _ext(ext), **found,
                                   "state": "ready", "created_at": _now()}
        return actual

    def add_audio(self, data: bytes | str | Path, ext: str | None = None, *, duration: float | None = None,
                  sample_rate: int | None = None, channels: int | None = None) -> str:
        """Put audio straight into the store (bytes, or a file's path); its sha256. `ext` defaults to the file's
        suffix, or .wav for bytes."""
        if isinstance(data, bytes):
            ext = ".wav" if ext is None else ext
            fd, tmp = tempfile.mkstemp(dir=self.store / "tmp")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            try:
                return self._put(Path(tmp), ext, duration=duration, sample_rate=sample_rate, channels=channels)
            finally:
                os.unlink(tmp)
        return self._put(Path(data), ext if ext is not None else Path(data).suffix, duration=duration,
                         sample_rate=sample_rate, channels=channels)

    def add_version(self, dataset_id: str, audio: Mapping[str, bytes], refs: Mapping[str, str] | None = None, *,
                    label_types: Iterable[str] = ("asr",), **manifest: Any) -> dict[str, Any]:
        """Store `audio` ({path: bytes}) and publish a version of it on top of the latest one, with `refs` as its
        refs/ files ({name: text}) and `manifest` as extra manifest fields (fake only, to set up a test). With no
        label_types, the version has audio only."""
        ds, types = self._ds(dataset_id), list(label_types)
        rows = []
        for path, data in audio.items():
            rec = self._audio[self.add_audio(data, PurePosixPath(path).suffix)]
            rows.append({"path": path, "sha256": rec["sha256"], "duration": rec["duration"] or 0.0,
                         "sample_rate": rec["sample_rate"] or 16000})
        m = {"format": FORMAT, "format_version": 2 if types else 3, "name": ds["name"], "created_at": _now(),
             "label_types": types, "counts": {"files": len(rows), "segments": 0,
                                              "hours": round(sum(r["duration"] for r in rows) / 3600, 6)},
             "audio_included": "none", **manifest}
        latest = self._latest(dataset_id)
        return self._publish_new(ds, m, rows, refs, latest["n"] if latest else None, False)

    def audio(self, sha256: str) -> dict[str, Any] | None:
        """The store's record of a file (fake only): sha256, size, ext, duration, sample_rate, channels, state."""
        return dict(self._audio[sha256]) if sha256 in self._audio else None

    def missing(self, sha256s: Iterable[str]) -> list[str]:
        shas = list(dict.fromkeys(sha256s))
        for s in shas:
            self._check_sha(s)
        return [s for s in shas if s not in self._audio]

    def held(self, app: str | None = None) -> set[str]:
        """The sha256s an app holds (fake only; this fake's app by default)."""
        return set(self._holds.get(app or self.app, {}))

    def hold(self, sha256s: Iterable[str]) -> int:
        shas = [self._check_sha(s) for s in dict.fromkeys(sha256s)]
        self._log("hold", tuple(shas))
        mine = self._holds.setdefault(self.app, {})
        for s in shas:
            mine.setdefault(s, _now())
        return len(mine)

    def release(self, sha256s: Iterable[str]) -> int:
        shas = [self._check_sha(s) for s in dict.fromkeys(sha256s)]
        self._log("release", tuple(shas))
        mine = self._holds.setdefault(self.app, {})
        for s in shas:
            mine.pop(s, None)
        return len(mine)

    def upload_audio(self, path: str | Path, sha256: str | None = None) -> str:
        path = Path(path)
        sha = sha256 or sha256_file(path)
        if sha in self._audio:
            return sha
        self._log("upload_audio", str(path))
        return self._put(path, path.suffix, sha)


__all__ = ["DatasetCatalog", "DatasetCatalogError", "FakeDatasetCatalog", "HasReferences", "InvalidVersion",
           "MissingAudio", "NotFound", "StaleParent", "check_folder", "digest", "sha256_file", "store_path"]
