"""Datasets and their versions (DATASET_CATALOG_PLAN.md §5, contracts/datasets/version.md).

A version is a reference package stored as a folder, without audio/ and outputs/:

    versions/<dataset_id>/<n>/manifest.json, files.csv, refs/<one references file>, refs/uids.jsonl,
                              refs/attributes.json

Rules (§5.3):

- A version never changes. Its digest is the sha256 of manifest.json (without the catalog's own fields),
  files.csv and the refs/ files, in that order (`digest`).
- A publish with the same digest as the dataset's latest version returns that version with 200.
- Every audio file must already be in the store, else 422 with the missing sha256s.
- The parent must be the latest version, else 409 naming the latest, unless `force`. A deleted parent is stale.
- Deleting a version removes its rows and folder. Its number is never used again (dataset.last_n), and a later
  version whose parent it was shows `parent_deleted`. Audio stays in the store until clean-up (gc.py).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

# The version format (check_folder, digest and the reading of references) is aicore.dataset_format, which
# FakeDatasetCatalog uses too. Callers still import check_folder and digest from here.
from aicore.dataset_format import (AUDIO_EXTS, FILES_HEADER, FORMAT, FORMAT_VERSION, INSTALL_ID, check_folder, digest,
                                   read_files_csv, rel_ok, version_files)

from ..db import DB, now_iso
from ..events import Events
from . import DatasetError, refs
from .store import COPY_CHUNK, AudioStore, check_sha

CATALOG_APP = "dataset-catalog"


def check_samples(folder: Path | None = None) -> list[str]:
    """Problems with the sample versions in contracts/datasets/samples (`just validate`): each one is a version
    as the catalog stores it, so it also has its source and export_version."""
    if folder is None:  # the platform checkout this file is in
        folder = Path(__file__).resolve().parents[2] / "contracts" / "datasets" / "samples"
    samples = sorted(p for p in folder.iterdir() if p.is_dir())
    problems = [] if samples else [f"no sample versions in {folder}"]
    for d in samples:
        found = check_folder(d)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").is_file() else {}
        if not found and (m.get("source") or {}).get("install_id") != INSTALL_ID:
            found.append(f"manifest.json: source.install_id must be {INSTALL_ID!r} in a stored version")
        problems += [f"{d.name}: {p}" for p in found]
    return problems


def unzip(zip_path: Path, dest: Path) -> Path:
    """Extract a package zip into `dest`; the package root, which is the zip's root or its only top-level folder."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise DatasetError(422, "the upload is not a zip", "invalid_zip") from None
    with zf:
        for info in zf.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if "__MACOSX" in parts or (parts and parts[-1] in (".DS_Store",)):
                continue
            if not rel_ok(name.rstrip("/")):
                raise DatasetError(422, f"the zip has an unsafe path {name!r}", "invalid_zip")
            target = dest.joinpath(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, COPY_CHUNK)
    top = [p for p in dest.iterdir()]
    if not (dest / "manifest.json").is_file() and len(top) == 1 and top[0].is_dir():
        return top[0]
    return dest


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "dataset"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FILES_HEADER, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


class _Sink(io.RawIOBase):
    """A write-only stream that hands its bytes to the caller as it goes: zipfile writes to it, the generator in
    `package` yields what it holds."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        self.buf += b
        return len(b)

    def take(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


# ----------------------------------------------------------------------------- datasets and versions

class Versions:
    def __init__(self, db: DB, store: AudioStore, events: Events):
        self.db = db
        self.store = store
        self.events = events
        self.root = store.root / "versions"

    def folder(self, dataset_id: str, n: int) -> Path:
        return self.root / dataset_id / str(n)

    def _event(self, dataset_id: str, n: int | None, action: str) -> None:
        self.events.publish("dataset", {"id": dataset_id, "n": n, "action": action})

    # ------------------------------------------------------------------------- records

    def _uses(self, dataset_id: str, n: int) -> list[dict[str, Any]]:
        return self.db.all("SELECT app, ref, created_at FROM use WHERE dataset_id = ? AND n = ? ORDER BY created_at, app, ref",
                           (dataset_id, n))

    def _version(self, row: dict[str, Any]) -> dict[str, Any]:
        parent = row["parent_n"]
        deleted = parent is not None and self.db.one("SELECT 1 AS x FROM version WHERE dataset_id = ? AND n = ?",
                                                     (row["dataset_id"], parent)) is None
        manifest = json.loads(row["manifest"])
        return {"dataset_id": row["dataset_id"], "n": row["n"], "parent_n": parent, "parent_deleted": deleted,
                "digest": row["digest"], "format_version": manifest.get("format_version"),
                "label_types": json.loads(row["label_types"]), "labels": json.loads(row["labels"]),
                "counts": json.loads(row["counts"]), "published_by": row["published_by"],
                "published_at": row["published_at"], "uses": self._uses(row["dataset_id"], row["n"]),
                "manifest": manifest}

    def _dataset_row(self, dataset_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM dataset WHERE id = ?", (dataset_id,))
        if row is None:
            raise DatasetError(404, f"no dataset {dataset_id!r}", "not_found")
        return row

    def _latest(self, dataset_id: str) -> dict[str, Any] | None:
        return self.db.one("SELECT * FROM version WHERE dataset_id = ? ORDER BY n DESC LIMIT 1", (dataset_id,))

    def _version_row(self, dataset_id: str, n: int) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM version WHERE dataset_id = ? AND n = ?", (dataset_id, n))
        if row is None:
            self._dataset_row(dataset_id)
            raise DatasetError(404, f"dataset {dataset_id} has no version {n}", "not_found")
        return row

    def _dataset(self, row: dict[str, Any], *, versions: bool = False) -> dict[str, Any]:
        latest = self._latest(row["id"])
        out = {"id": row["id"], "name": row["name"], "language": row["language"], "description": row["description"],
               "tags": json.loads(row["tags"]), "created_by": row["created_by"], "created_at": row["created_at"],
               "updated_at": row["updated_at"], "latest": self._version(latest) if latest else None}
        if versions:
            out["versions"] = [self._version(r) for r in
                               self.db.all("SELECT * FROM version WHERE dataset_id = ? ORDER BY n DESC", (row["id"],))]
        return out

    def get(self, dataset_id: str) -> dict[str, Any]:
        return self._dataset(self._dataset_row(dataset_id), versions=True)

    def version(self, dataset_id: str, n: int) -> dict[str, Any]:
        return self._version(self._version_row(dataset_id, n))

    def list(self, q: str | None = None, language: str | None = None, label_type: str | None = None,
             app: str | None = None) -> list[dict[str, Any]]:
        out = []
        for row in self.db.all("SELECT * FROM dataset ORDER BY updated_at DESC, name"):
            if q and q.lower() not in " ".join([row["name"], row["description"] or "", row["tags"]]).lower():
                continue
            if language and (row["language"] or "").lower() != language.lower():
                continue
            if app and row["created_by"] != app and not self.db.one(
                    "SELECT 1 AS x FROM version WHERE dataset_id = ? AND published_by = ? "
                    "UNION SELECT 1 FROM use WHERE dataset_id = ? AND app = ?", (row["id"], app, row["id"], app)):
                continue
            d = self._dataset(row)
            if label_type and label_type not in ((d["latest"] or {}).get("label_types") or []):
                continue
            out.append(d)
        return out

    # ------------------------------------------------------------------------- datasets

    def create(self, name: str, app: str, language: str | None = None, description: str | None = None,
               tags: list[str] | None = None) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise DatasetError(422, "name is required", "invalid_request")
        tags = [str(t) for t in tags] if isinstance(tags, list) else []
        did, now = uuid.uuid4().hex, now_iso()
        self.db.execute("INSERT INTO dataset (id, name, language, description, tags, created_by, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (did, name, language or None, description or None, json.dumps(sorted(set(tags))),
                         app or None, now, now))
        self._event(did, None, "created")
        return self.get(did)

    def update(self, dataset_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        self._dataset_row(dataset_id)
        sets: dict[str, Any] = {}
        if "name" in fields:
            if not (fields["name"] or "").strip():
                raise DatasetError(422, "name can't be empty", "invalid_request")
            sets["name"] = fields["name"].strip()
        if "description" in fields:
            sets["description"] = fields["description"] or None
        if "tags" in fields:
            sets["tags"] = json.dumps(sorted(set(fields["tags"] or [])))
        if sets:
            sets["updated_at"] = now_iso()
            self.db.execute(f"UPDATE dataset SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?",
                            (*sets.values(), dataset_id))
            self._event(dataset_id, None, "updated")
        return self.get(dataset_id)

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset with all its versions. Its audio stays in the store until clean-up."""
        with self.store.lock:
            self._dataset_row(dataset_id)
            with self.db.tx() as c:
                for table in ("version_file", "use", "version"):
                    c.execute(f"DELETE FROM {table} WHERE dataset_id = ?", (dataset_id,))
                c.execute("DELETE FROM dataset WHERE id = ?", (dataset_id,))
            self._remove_folder(self.root / dataset_id)
        self._event(dataset_id, None, "deleted")

    def _remove_folder(self, folder: Path) -> None:
        """Move a folder out of versions/ first, so a reader of the mount sees it whole or not at all."""
        if folder.exists():
            gone = self.store.root / "tmp" / f"deleted-{uuid.uuid4().hex}"
            os.replace(folder, gone)
            shutil.rmtree(gone, ignore_errors=True)

    # ------------------------------------------------------------------------- publishing

    def publish(self, dataset_id: str, zip_path: Path, parent: int | None, app: str,
                force: bool = False) -> tuple[dict[str, Any], bool]:
        """Publish a zip of a version folder (or a whole reference package). (version, created): created is False
        when the digest matched the latest version, which is returned instead."""
        ds = self._dataset_row(dataset_id)
        work = self.store.scratch()
        try:
            root = unzip(zip_path, work)
            return self._publish(ds, root, parent, app, force)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _read_folder(self, root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
        problems = check_folder(root, package=True, hash_audio=False)
        if problems:
            raise DatasetError(422, "the version is invalid: " + "; ".join(problems[:10]), "invalid_version",
                               problems=problems)
        return json.loads((root / "manifest.json").read_text(encoding="utf-8")), read_files_csv(root)

    def _store_included_audio(self, root: Path, rows: list[dict[str, str]]) -> None:
        """Move a package's audio/<path> files into the store, checking each against its sha256. The folder is then a
        version: no audio/ and audio_included 'none' (a catalog field, so the digest doesn't change)."""
        if not (root / "audio").is_dir():
            return
        for r in rows:
            src = root / "audio" / r["path"]
            if src.is_file():  # the package was unzipped under the store's tmp/, so put_file can move it
                self.store.put_file(src, PurePosixPath(r["path"]).suffix, r["sha256"])
        shutil.rmtree(root / "audio")
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        (root / "manifest.json").write_text(_json({**m, "audio_included": "none"}), encoding="utf-8")

    def _missing(self, rows: list[dict[str, str]]) -> list[str]:
        have = self.store.ready(r["sha256"] for r in rows)
        return list(dict.fromkeys(r["sha256"] for r in rows if r["sha256"] not in have))

    def _check_parent(self, ds: dict[str, Any], latest: dict[str, Any] | None, parent: int | None,
                      force: bool) -> None:
        if parent is not None and not 1 <= parent <= ds["last_n"]:
            raise DatasetError(422, f"dataset {ds['id']} never had a version {parent}", "invalid_request")
        latest_n = latest["n"] if latest else None
        if parent == latest_n or force:
            return
        if parent is not None and self.db.one("SELECT 1 AS x FROM version WHERE dataset_id = ? AND n = ?",
                                              (ds["id"], parent)) is None:
            msg = f"v{parent} was deleted"
        else:
            msg = f"the version is based on {f'v{parent}' if parent else 'no version'}"
        msg += f", but the latest version is {f'v{latest_n}' if latest_n else 'none'}; publish with force to keep it"
        raise DatasetError(409, msg, "stale_parent", latest=self._version(latest) if latest else None)

    def _publish(self, ds: dict[str, Any], root: Path, parent: int | None, app: str, force: bool,
                 *, check_parent: bool = True) -> tuple[dict[str, Any], bool]:
        app = (app or "").strip() or CATALOG_APP
        manifest, rows = self._read_folder(root)
        self._store_included_audio(root, rows)
        d = digest(root)
        with self.store.lock:
            ds = self._dataset_row(ds["id"])
            latest = self._latest(ds["id"])
            if latest and latest["digest"] == d:
                return self._version(latest), False
            if check_parent:
                self._check_parent(ds, latest, parent, force)
            missing = self._missing(rows)
            if missing:
                raise DatasetError(422, f"{len(missing)} audio files are not in the store", "missing_audio",
                                   missing=missing)
            n = ds["last_n"] + 1
            now = now_iso()
            manifest.update({"source": {"app": app, "install_id": INSTALL_ID, "dataset_uid": ds["id"],
                                        "dataset_name": ds["name"]},
                             "export_version": n, "created_at": now, "audio_included": "none"})
            parsed_labels = self._write_folder(ds["id"], n, root, manifest)
            counts = manifest["counts"]
            with self.db.tx() as c:
                c.execute("INSERT INTO version (dataset_id, n, parent_n, digest, manifest, counts, label_types, labels, "
                          "published_by, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (ds["id"], n, parent, d, json.dumps(manifest), json.dumps(counts),
                           json.dumps(manifest["label_types"]), json.dumps(parsed_labels), app, now))
                c.executemany("INSERT INTO version_file (dataset_id, n, path, sha256, duration, sample_rate) "
                              "VALUES (?, ?, ?, ?, ?, ?)",
                              [(ds["id"], n, r["path"], r["sha256"], float(r["duration"]), int(r["sample_rate"]))
                               for r in rows])
                c.execute("UPDATE dataset SET last_n = ?, updated_at = ? WHERE id = ?", (n, now, ds["id"]))
            out = self._version(self._version_row(ds["id"], n))
        self._event(ds["id"], n, "published")
        return out, True

    def _write_folder(self, dataset_id: str, n: int, src: Path, manifest: dict[str, Any]) -> list[str]:
        """Write versions/<id>/<n>/ from a checked folder, through a temporary folder renamed into place. The label
        names its references carry."""
        tmp = self.store.root / "tmp" / f"version-{uuid.uuid4().hex}"
        (tmp / "refs").mkdir(parents=True)
        try:
            (tmp / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            shutil.copyfile(src / "files.csv", tmp / "files.csv")
            if (src / "refs").is_dir():
                for p in (src / "refs").iterdir():
                    shutil.copyfile(p, tmp / "refs" / p.name)
            if not any((tmp / "refs").iterdir()):
                (tmp / "refs").rmdir()
            labels = refs.parse(tmp)["labels"]
            dest = self.folder(dataset_id, n)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return labels

    def import_package(self, zip_path: Path, dataset_id: str | None, app: str, filename: str = "",
                       preview: bool = False) -> tuple[dict[str, Any], bool]:
        """A reference package zip as a new dataset, or as the next version of `dataset_id` (based on its latest
        version). With `preview`, nothing is stored: the manifest, problems and missing audio are returned."""
        ds = self._dataset_row(dataset_id) if dataset_id else None
        work = self.store.scratch()
        try:
            root = unzip(zip_path, work)
            if preview:
                return self._preview(ds, root), False
            manifest, rows = self._read_folder(root)
            self._store_included_audio(root, rows)  # before the dataset is made, so bad audio makes none
            missing = self._missing(rows)
            if missing:
                raise DatasetError(422, f"{len(missing)} audio files are neither in the package nor in the store",
                                   "missing_audio", missing=missing)
            made = ds is None
            if ds is None:
                ds = self._dataset_row(self.create(manifest.get("name") or Path(filename).stem or "Imported", app,
                                                   manifest.get("language"), manifest.get("description"),
                                                   manifest.get("tags"))["id"])
            latest = self._latest(ds["id"])
            try:
                return self._publish(ds, root, latest["n"] if latest else None, app, False, check_parent=False)
            except Exception:
                if made:  # an import makes its dataset with its first version, or not at all
                    self.delete(ds["id"])
                raise
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _preview(self, ds: dict[str, Any] | None, root: Path) -> dict[str, Any]:
        problems = check_folder(root, package=True)
        if problems:
            return {"ok": False, "problems": problems, "manifest": None, "files": 0, "missing": [],
                    "same_as_latest": False}
        manifest, rows = json.loads((root / "manifest.json").read_text(encoding="utf-8")), read_files_csv(root)
        included = {r["sha256"] for r in rows if (root / "audio" / r["path"]).is_file()}
        latest = self._latest(ds["id"]) if ds else None
        return {"ok": True, "problems": [], "manifest": manifest, "files": len(rows),
                "missing": [s for s in self._missing(rows) if s not in included],
                "same_as_latest": bool(latest and latest["digest"] == digest(root))}

    def publish_audio(self, dataset_id: str, files: list[dict[str, str]], parent: int | None, app: str,
                      force: bool = False) -> tuple[dict[str, Any], bool]:
        """A version with audio only (format version 3, label_types []): the parent version's files plus `files`.
        409 when the latest version has references: those datasets get new audio through Label."""
        ds = self._dataset_row(dataset_id)
        latest = self._latest(dataset_id)
        if latest and json.loads(latest["label_types"]):
            raise DatasetError(409, f"v{latest['n']} has references: add audio to this dataset through an app",
                               "has_references", latest=self._version(latest))
        base = self.db.all("SELECT path, sha256, duration, sample_rate FROM version_file WHERE dataset_id = ? AND n = ? "
                           "ORDER BY path", (dataset_id, latest["n"])) if latest else []
        by_path = {r["path"]: r for r in base}
        stems = {PurePosixPath(p).stem.lower(): p for p in by_path}
        audio = self.store.ready(check_sha(f.get("sha256", "")) for f in files)
        missing, unreadable = [], []
        for f in files:
            path, sha = f.get("path") or "", f["sha256"]
            if not rel_ok(path) or PurePosixPath(path).suffix.lower() not in AUDIO_EXTS:
                raise DatasetError(422, f"bad path {path!r}", "invalid_request")
            if path in by_path:
                if by_path[path]["sha256"] != sha:
                    raise DatasetError(422, f"{path} is already in the dataset with other audio", "invalid_request")
                continue
            stem = PurePosixPath(path).stem.lower()
            if stem in stems:
                raise DatasetError(422, f"the stem of {path} repeats {stems[stem]}", "invalid_request")
            if sha not in audio:
                missing.append(sha)
                continue
            rec = audio[sha]
            if rec["duration"] is None or rec["sample_rate"] is None:
                unreadable.append(path)
                continue
            stems[stem] = path
            by_path[path] = {"path": path, "sha256": sha, "duration": rec["duration"], "sample_rate": rec["sample_rate"]}
        if missing:
            raise DatasetError(422, f"{len(missing)} audio files are not in the store", "missing_audio", missing=missing)
        if unreadable:
            raise DatasetError(422, f"ffprobe can't read {len(unreadable)} files", "unreadable_audio",
                               unreadable=unreadable)
        rows = sorted(by_path.values(), key=lambda r: r["path"])
        manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "name": ds["name"], "created_at": now_iso(),
                    **{k: ds[k] for k in ("language", "description") if ds[k]},
                    "tags": json.loads(ds["tags"]), "label_types": [],
                    "counts": {"files": len(rows), "segments": 0,
                               "hours": round(sum(float(r["duration"]) for r in rows) / 3600, 6)},
                    "audio_included": "none"}
        work = self.store.scratch()
        try:
            (work / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            _write_csv(work / "files.csv", rows)
            return self._publish(ds, work, parent, app, force)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def delete_version(self, dataset_id: str, n: int) -> None:
        """Delete one version. Its number is never used again; its audio stays until clean-up."""
        with self.store.lock:
            self._version_row(dataset_id, n)
            with self.db.tx() as c:
                for table in ("version_file", "use", "version"):
                    c.execute(f"DELETE FROM {table} WHERE dataset_id = ? AND n = ?", (dataset_id, n))
                c.execute("UPDATE dataset SET updated_at = ? WHERE id = ?", (now_iso(), dataset_id))
            self._remove_folder(self.folder(dataset_id, n))
        self._event(dataset_id, n, "deleted")

    # ------------------------------------------------------------------------- uses

    def add_use(self, dataset_id: str, n: int, app: str, ref: str) -> None:
        self._version_row(dataset_id, n)
        if not app.strip() or not ref.strip():
            raise DatasetError(422, "app and ref are required", "invalid_request")
        self.db.execute("INSERT OR IGNORE INTO use (dataset_id, n, app, ref, created_at) VALUES (?, ?, ?, ?, ?)",
                        (dataset_id, n, app.strip(), ref.strip(), now_iso()))

    def remove_use(self, dataset_id: str, n: int, app: str, ref: str) -> None:
        """Idempotent, and it works after the version was deleted, when its uses went with it. app and ref are
        stripped, as add_use stores them."""
        self.db.execute("DELETE FROM use WHERE dataset_id = ? AND n = ? AND app = ? AND ref = ?",
                        (dataset_id, n, app.strip(), ref.strip()))

    # ------------------------------------------------------------------------- reading a version

    def _parsed(self, row: dict[str, Any]) -> dict[str, Any]:
        return refs.read(self.folder(row["dataset_id"], row["n"]), row["digest"])

    def _rows(self, dataset_id: str, n: int) -> dict[str, dict[str, Any]]:
        return {r["path"]: r for r in self.db.all("SELECT path, sha256, duration, sample_rate FROM version_file "
                                                  "WHERE dataset_id = ? AND n = ? ORDER BY path", (dataset_id, n))}

    def files(self, dataset_id: str, n: int, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        row = self._version_row(dataset_id, n)
        parsed = self._parsed(row)
        total = self.db.one("SELECT COUNT(*) AS n FROM version_file WHERE dataset_id = ? AND n = ?",
                            (dataset_id, n))["n"]  # type: ignore[index]
        page = self.db.all("SELECT path, sha256, duration, sample_rate FROM version_file WHERE dataset_id = ? AND n = ? "
                           "ORDER BY path LIMIT ? OFFSET ?", (dataset_id, n, limit, offset))
        return {"total": total, "offset": offset, "limit": limit,
                "files": [{**r, **refs.file_summary(parsed, r["path"])} for r in page]}

    def refs(self, dataset_id: str, n: int, path: str) -> dict[str, Any]:
        row = self._version_row(dataset_id, n)
        if not self.db.one("SELECT 1 AS x FROM version_file WHERE dataset_id = ? AND n = ? AND path = ?",
                           (dataset_id, n, path)):
            raise DatasetError(404, f"v{n} has no file {path!r}", "not_found")
        return refs.file_refs(self._parsed(row), path)

    def compare(self, dataset_id: str, a: int, b: int) -> dict[str, Any]:
        ra, rb = self._version_row(dataset_id, a), self._version_row(dataset_id, b)
        return {"a": a, "b": b, **refs.compare(self._rows(dataset_id, a), self._rows(dataset_id, b),
                                               self._parsed(ra), self._parsed(rb))}

    def package(self, dataset_id: str, n: int, audio: bool) -> tuple[str, Iterator[bytes]]:
        """The version as a reference package zip, streamed as it is written: (file name, chunks). With `audio`,
        every original goes under audio/<path> and audio_included is 'all'."""
        row = self._version_row(dataset_id, n)
        ds = self._dataset_row(dataset_id)
        folder = self.folder(dataset_id, n)
        top = f"{_slug(ds['name'])}-v{n}"
        manifest = {**json.loads(row["manifest"]), "audio_included": "all" if audio else "none"}
        texts = [("manifest.json", _json(manifest).encode())] + \
                [(name, p.read_bytes()) for name, p in version_files(folder) if name != "manifest.json"]
        files = list(self._rows(dataset_id, n).values()) if audio else []
        for f in files:
            self.store.require(f["sha256"])

        def chunks() -> Iterator[bytes]:
            sink = _Sink()
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in texts:
                    zf.writestr(f"{top}/{name}", data)
                    yield sink.take()
                for f in files:
                    info = zipfile.ZipInfo(f"{top}/audio/{f['path']}", date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED  # audio is compressed already, or not worth it
                    with open(self.store.path(f["sha256"]), "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                        for chunk in iter(lambda: src.read(COPY_CHUNK), b""):
                            dst.write(chunk)
                            if len(sink.buf) >= COPY_CHUNK:
                                yield sink.take()
                    yield sink.take()
            yield sink.take()

        return f"{top}.zip", chunks()
