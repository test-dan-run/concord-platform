"""Clean-up of the audio store (DATASET_CATALOG_PLAN.md §4.3).

An audio file is deleted only when no version lists it, no app holds it, and it has been in the store for more
than 24 hours, so an upload that hasn't been published or held yet is safe. A partial upload counts from its last
write. Clean-up runs only when a person starts it, from the Storage page or with `just datasets-gc` (`main` below),
and both show a dry run first.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import httpx

from aicore.catalog import DEFAULT_URL, catalog_url

from ..db import iso
from . import DatasetError
from .store import AudioStore

GRACE_S = 24 * 3600.0
SAMPLE = 20


def candidates(store: AudioStore, now: float | None = None) -> list[dict[str, Any]]:
    """The audio records clean-up would delete, oldest first."""
    now = time.time() if now is None else now
    cutoff = iso(now - GRACE_S)
    rows = store.db.all("SELECT * FROM audio a WHERE "
                        "NOT EXISTS (SELECT 1 FROM version_file v WHERE v.sha256 = a.sha256) AND "
                        "NOT EXISTS (SELECT 1 FROM hold h WHERE h.sha256 = a.sha256) AND a.created_at < ? "
                        "ORDER BY a.created_at, a.sha256", (cutoff,))
    out = []
    for r in rows:
        if r["state"] == "partial":
            tmp = store.tmp_path(r["sha256"])
            if store.busy(r["sha256"]) or (tmp.is_file() and tmp.stat().st_mtime > now - GRACE_S):
                continue
            r["size"] = tmp.stat().st_size if tmp.is_file() else 0
        out.append(r)
    return out


def collect(store: AudioStore, dry_run: bool = True, *, expect_files: int | None = None) -> dict[str, Any]:
    """{"dry_run", "files", "bytes", "sample"}: what clean-up deletes, or would delete on a dry run. The rules are
    checked again under the store's lock, so a publish running at the same time keeps its audio. With
    `expect_files` (the count a dry run showed), a real run that finds more deletes nothing: 409 gc_changed."""
    with store.lock:
        found = candidates(store)
        if not dry_run:
            if expect_files is not None and len(found) > expect_files:
                raise DatasetError(409, "clean-up found more files than the check showed: check again", "gc_changed")
            for r in found:
                store.remove(r["sha256"])
    return {"dry_run": dry_run, "files": len(found), "bytes": sum(r["size"] for r in found),
            "sample": [{k: r[k] for k in ("sha256", "size", "ext", "state", "created_at")} for r in found[:SAMPLE]]}


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    """`just datasets-gc`: a dry run through the running catalog service, then the clean-up if the person agrees."""
    url = (catalog_url() or DEFAULT_URL).rstrip("/")
    try:
        dry = httpx.post(f"{url}/api/store/gc", json={"dry_run": True}, timeout=600).raise_for_status().json()
    except httpx.HTTPError as exc:
        print(f"the catalog service at {url} did not answer: {exc}", file=sys.stderr)
        return 1
    print(f"Clean-up would delete {dry['files']} audio files ({_size(dry['bytes'])}) that no version lists, "
          "no app holds, and that are older than 24 hours.")
    for r in dry["sample"]:
        print(f"  {r['sha256']}{r['ext']}  {_size(r['size'])}  {r['created_at']}")
    if dry["files"] > len(dry["sample"]):
        print(f"  ... and {dry['files'] - len(dry['sample'])} more")
    if not dry["files"]:
        return 0
    if input("Delete them? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing was deleted.")
        return 0
    done = httpx.post(f"{url}/api/store/gc", json={"dry_run": False}, timeout=600).raise_for_status().json()
    print(f"Deleted {done['files']} audio files ({_size(done['bytes'])}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
