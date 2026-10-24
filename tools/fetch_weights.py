"""Download a custom service's weights into its manifest `weights.path` and verify sha256.

By convention new services use ${AI_ASSETS}/custom-models/<name>/<version>; a manifest may instead
point at a copy that already exists elsewhere under ${AI_ASSETS}.

    python tools/fetch_weights.py NAME            # download missing/mismatched files, verify all
    python tools/fetch_weights.py NAME --verify   # only check what is on disk

Sources and checksums come from custom/<name>/weights.yaml. HF_TOKEN is sent to huggingface.co only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from tools.schema import Platform, PlatformError, load_platform, load_yaml

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def weights_spec(p: Platform, name: str) -> tuple[dict, Path]:
    s = p.get(name)
    if not s.is_custom or s.build is None:
        raise SystemExit(f"{s.name} is an engine service; its weights are the local copy {s.model} in ${{AI_ASSETS}} "
                         "(`hf download <repo> --revision <sha> --local-dir $AI_ASSETS/<dir>`, plus a SOURCE.txt)")
    f = p.root / s.build.context / "weights.yaml"
    if not f.is_file():
        raise SystemExit(f"{s.name} has no weights.yaml (nothing to fetch)")
    spec = load_yaml(f)
    return spec, p.weights_dir(s)


def download(url: str, target: Path, expected: str) -> None:
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token and urlparse(url).hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"  # httpx drops it on cross-host redirects (CDN)
    tmp = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_bytes(CHUNK):
                fh.write(block)
                h.update(block)
                done += len(block)
                if total > 50 * CHUNK and sys.stdout.isatty():
                    print(f"\r    {done / CHUNK:.0f}/{total / CHUNK:.0f} MB", end="", flush=True)
    if total > 50 * CHUNK and sys.stdout.isatty():
        print()
    if h.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch for {url}: got {h.hexdigest()}, expected {expected}")
    tmp.replace(target)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--verify", action="store_true", help="only verify files already on disk")
    args = ap.parse_args(argv)
    try:
        p = load_platform()
        spec, dest = weights_spec(p, args.name)
    except PlatformError as exc:
        print("validation FAILED:\n  - " + "\n  - ".join(exc.errors), file=sys.stderr)
        return 1
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    print(f"{args.name} {spec['version']} -> {dest}")
    bad = 0
    for f in spec["files"]:
        target = dest / f["path"]
        if target.is_file() and sha256_of(target) == f["sha256"]:
            print(f"  ok        {f['path']}")
            continue
        if args.verify:
            print(f"  {'MISMATCH' if target.exists() else 'MISSING':<9} {f['path']}")
            bad += 1
            continue
        print(f"  download  {f['path']} ({f.get('size', 0) / 1e6:.1f} MB)", flush=True)
        download(f["url"], target, f["sha256"])
    if bad:
        print(f"{bad} file(s) missing or wrong; run without --verify to fetch")
        return 1
    print("all files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
