"""`python -m catalog import <bundle>`: load an export bundle (catalog/exporter.py) on this host.

1. Images: `docker load -i images.tar`, or for a registry export `docker pull` each pushed ref and tag it
   back to the manifest's image ref.
2. Manifests: each `models/<id>/service.yaml` is written into platform/services (an existing manifest that
   differs is left alone unless --force). The platform must still load afterwards, or nothing is kept.
3. Weights: `models/<id>/weights/` is copied to where the manifest mounts weights from (under $AI_ASSETS).
4. Registration: display name, tags and "derived from" are registered through the running catalog
   (AICORE_CATALOG, else http://127.0.0.1:8700), which queues verification; without a running catalog
   they are written to its database and verified on its next start.

The target re-verifies every build. A build key that matches one the target already knows (same image
id, config and weights) keeps the outputs indexed under it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from tools import identity
from tools.schema import PLATFORM_DIR, PlatformError, load_platform

Run = Callable[..., subprocess.CompletedProcess]


class BundleError(Exception):
    pass


def _say(msg: str) -> None:
    print(msg, flush=True)


def open_bundle(path: Path) -> Path:
    """The bundle folder: the path itself, or a .tar extracted next to it (once)."""
    if path.is_dir():
        return path
    if not path.is_file():
        raise BundleError(f"{path} is neither a bundle folder nor a .tar")
    with tarfile.open(path) as tar:
        top = {m.name.split("/", 1)[0] for m in tar.getmembers()}
        if len(top) != 1:
            raise BundleError(f"{path} is not a catalog export (expected one top-level folder, found {sorted(top)})")
        root = path.parent / top.pop()
        if not root.exists():
            _say(f"extracting {path.name} into {root}")
            tar.extractall(path.parent, filter="data")
    return root


def load_images(root: Path, manifest: dict[str, Any], run: Run) -> None:
    if manifest.get("images"):
        _say(f"loading {manifest['images']}")
        r = run(["docker", "load", "-i", str(root / manifest["images"])], capture_output=True, text=True)
        if r.returncode != 0:
            raise BundleError(f"docker load failed: {r.stderr.strip()[-400:]}")
        return
    for m in manifest["models"]:
        pushed = (m.get("pushed") or {}).get("ref")
        if not pushed:
            continue
        _say(f"pulling {pushed}")
        for cmd in (["docker", "pull", pushed], ["docker", "tag", pushed, m["image"]]):
            r = run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise BundleError(f"{' '.join(cmd[:2])} {pushed} failed: {r.stderr.strip()[-400:]}")


def write_manifests(root: Path, manifest: dict[str, Any], platform_dir: Path, force: bool) -> list[str]:
    """Write each service.yaml into services/; returns the ids written. Rolls back if the platform no longer loads."""
    services = platform_dir / "services"
    try:
        current = load_platform(platform_dir)
        known = dict(current.files)
    except PlatformError:
        known = {}
    written: list[tuple[Path, bytes | None]] = []
    ids = []
    for m in manifest["models"]:
        src = root / "models" / m["id"] / "service.yaml"
        dst = known.get(m["id"]) or services / f"{m['id']}.yaml"
        new = src.read_bytes()
        old = dst.read_bytes() if dst.is_file() else None
        if old == new:
            _say(f"{m['id']}: manifest already matches")
            ids.append(m["id"])
            continue
        if old is not None and not force:
            _say(f"{m['id']}: {dst} differs from the bundle's manifest; kept (use --force to overwrite)")
            continue
        dst.write_bytes(new)
        written.append((dst, old))
        ids.append(m["id"])
        _say(f"{m['id']}: wrote {dst}")
    try:
        load_platform(platform_dir)
    except PlatformError as exc:
        for dst, old in written:
            if old is None:
                dst.unlink()
            else:
                dst.write_bytes(old)
        raise BundleError("the platform does not load with the bundle's manifests, so none were kept:\n  "
                           + "\n  ".join(exc.errors)) from None
    return ids


def load_dotenv(platform_dir: Path) -> None:
    """Variables from platform/.env that the environment does not set (as `just` loads them): AI_ASSETS above all."""
    env = platform_dir / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v


def copy_weights(root: Path, manifest: dict[str, Any], platform_dir: Path, ids: list[str]) -> None:
    p = load_platform(platform_dir)
    for m in manifest["models"]:
        src = root / "models" / m["id"] / "weights"
        if m["id"] not in ids or not src.is_dir():
            if (m.get("weights") or {}).get("note"):
                _say(f"{m['id']}: weights {m['weights']['note']}")
            continue
        s = p.get(m["id"])
        ws = identity.weights_source(p, s)
        dst = ws.path if ws.kind == "dir" else p.weights_dir(s)
        if dst is None:
            _say(f"{m['id']}: the manifest mounts no weights; skipped {src}")
            continue
        n = 0
        files = identity.weight_files(src)
        # SOURCE.txt is the model card's provenance: copied, but never part of the weights hash
        files += [(name, src / name) for name in sorted(identity.IGNORED_FILES) if (src / name).is_file()]
        for rel, f in files:
            target = dst / rel
            if target.is_file() and target.stat().st_size == f.stat().st_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
        _say(f"{m['id']}: {n} weights file(s) copied to {dst}")


def register(root: Path, manifest: dict[str, Any], ids: list[str], catalog: str | None, db_path: Path) -> None:
    regs = []
    for m in manifest["models"]:
        f = root / "models" / m["id"] / "registration.json"
        reg = json.loads(f.read_text()) if f.is_file() else {"registered": True}
        if m["id"] in ids and reg.get("registered", True):
            regs.append({"id": m["id"], "display_name": reg.get("display_name"), "tags": reg.get("tags") or [],
                         "parent_id": reg.get("parent_id")})
    if not regs:
        return
    if catalog:
        try:
            with httpx.Client(base_url=catalog, timeout=30) as c:
                c.get("/api/health").raise_for_status()
                for r in regs:  # parents first are not guaranteed: register, then set parents that exist
                    c.post(f"/api/models/{r['id']}/register", json={"display_name": r["display_name"],
                                                                    "tags": r["tags"]}).raise_for_status()
                for r in regs:
                    if r["parent_id"]:
                        c.patch(f"/api/models/{r['id']}", json={"parent_id": r["parent_id"]})
            _say(f"registered {len(regs)} model(s) with the catalog at {catalog}; it verifies each build now")
            return
        except httpx.HTTPError as exc:
            _say(f"the catalog at {catalog} is not answering ({exc}); writing registrations to {db_path}")
    from .db import DB, now_iso

    db = DB(db_path)
    try:
        for r in regs:
            db.execute("INSERT INTO registration (model_id, display_name, tags, parent_id, registered_at) VALUES (?, ?, ?, ?, ?) "
                       "ON CONFLICT(model_id) DO UPDATE SET display_name = excluded.display_name, tags = excluded.tags, "
                       "parent_id = excluded.parent_id",
                       (r["id"], r["display_name"], json.dumps(r["tags"]), r["parent_id"], now_iso()))
    finally:
        db.close()
    _say(f"registered {len(regs)} model(s) in {db_path}; the catalog verifies them when it starts")


def import_bundle(path: Path, platform_dir: Path = PLATFORM_DIR, catalog: str | None = None,
                  db_path: Path | None = None, force: bool = False, images: bool = True, run: Run = subprocess.run) -> list[str]:
    root = open_bundle(path)
    mf = root / "manifest.json"
    if not mf.is_file():
        raise BundleError(f"{root} has no manifest.json: not a catalog export")
    manifest = json.loads(mf.read_text())
    if manifest.get("format") != 1:
        raise BundleError(f"bundle format {manifest.get('format')!r} is not supported (this catalog reads format 1)")
    _say(f"importing {len(manifest['models'])} model(s) from {root.name} (exported {manifest.get('created_at')})")
    if any((root / "models" / m["id"] / "weights").is_dir() for m in manifest["models"]):
        try:
            assets = load_platform(platform_dir).config.assets_path()
        except PlatformError as exc:
            raise BundleError("the platform does not load:\n  " + "\n  ".join(exc.errors)) from None
        if not assets.is_dir():  # checked first: never write manifests, or a stray weights tree, for a wrong path
            raise BundleError(f"AI_ASSETS is {assets}, which does not exist; set AI_ASSETS (platform/.env) to this "
                              "host's weights folder, then import again")
    if images:
        load_images(root, manifest, run)
    ids = write_manifests(root, manifest, platform_dir, force)
    copy_weights(root, manifest, platform_dir, ids)
    from .settings import Settings

    register(root, manifest, ids, catalog, db_path or Settings().db_path)
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m catalog import", description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", type=Path, help="an export folder, or its .tar from the Export page")
    ap.add_argument("--catalog", default=os.environ.get("AICORE_CATALOG") or "http://127.0.0.1:8700",
                    help="the running catalog to register with (default: AICORE_CATALOG or http://127.0.0.1:8700)")
    ap.add_argument("--platform-dir", type=Path, default=PLATFORM_DIR)
    ap.add_argument("--force", action="store_true", help="overwrite manifests that differ from the bundle's")
    ap.add_argument("--no-images", action="store_true", help="skip docker load / pull (images already present)")
    args = ap.parse_args(argv)
    load_dotenv(args.platform_dir)
    try:
        import_bundle(args.bundle, args.platform_dir, args.catalog.rstrip("/") or None, force=args.force,
                      images=not args.no_images)
    except BundleError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    return 0
