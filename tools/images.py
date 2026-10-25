"""Build and push custom service images (and their shared base).

    python tools/images.py build NAME [--rebuild-base]
    python tools/images.py push NAME
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import Manifest, Platform, PlatformError, load_bases, load_platform


DIGEST_LABEL = "aiplatform.base-digest"


def image_exists(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref], capture_output=True).returncode == 0


def image_label(ref: str, label: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", ref],
                       capture_output=True, text=True)
    return (json.loads(r.stdout or "null") or {}).get(label) if r.returncode == 0 else None


def _ignore_rules(context: Path) -> list[tuple[bool, re.Pattern]]:
    """The context's .dockerignore as (re-include, regex) pairs: ** spans folders, * and ? stay in one."""
    f = context / ".dockerignore"
    rules = []
    for line in f.read_text().splitlines() if f.is_file() else []:
        pat = line.strip()
        if pat and not pat.startswith("#"):
            parts = re.split(r"(\*\*/|\*\*|\*|\?)", pat.removeprefix("!").strip("/"))
            rx = "".join({"**/": "(.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(s, re.escape(s)) for s in parts)
            rules.append((pat.startswith("!"), re.compile(rx)))
    return rules


def _ignored(rel: str, rules: list[tuple[bool, re.Pattern]]) -> bool:
    """Docker's rule: the last pattern that matches the path or one of its parent folders decides."""
    paths = [rel.rsplit("/", n)[0] for n in range(rel.count("/") + 1)]
    out = False
    for keep, rx in rules:
        if any(rx.fullmatch(p) for p in paths):
            out = not keep
    return out


def base_digest(bases_dir: Path, spec: dict) -> str:
    """Hash of what a base image is built from: its Dockerfile, build args, the .dockerignore and the
    context files its COPY lines read, less what .dockerignore leaves out. Other bases' files do not count."""
    h = hashlib.sha256(json.dumps({"dockerfile": spec["dockerfile"], "args": spec.get("args") or {}}, sort_keys=True).encode())
    dockerfile = bases_dir / spec["dockerfile"]
    rules = _ignore_rules(bases_dir)
    files = {dockerfile, bases_dir / ".dockerignore"}
    for line in dockerfile.read_text().splitlines():
        words = line.split()
        if words[:1] in (["COPY"], ["ADD"]):
            for src in (bases_dir / w for w in words[1:-1] if not w.startswith("--")):
                files |= {f for f in [src, *src.rglob("*")]
                          if f.is_file() and not _ignored(f.relative_to(bases_dir).as_posix(), rules)}
    for f in sorted(f for f in files if f.is_file()):
        h.update(str(f.relative_to(bases_dir)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_base(p: Platform, s: Manifest, force: bool = False) -> str | None:
    """Build the service's base image if missing, stale (its inputs changed) or forced.

    An older version a service pins (bases.yaml older_versions) is never stale: its files are gone, so
    an existing image is kept as it is, and a missing one is built from the current files.
    """
    ref = p.base_image(s)
    if ref is None:
        return None
    bases_dir = p.root / "custom" / "_bases"
    spec = load_bases(p.root)[s.base]
    digest = base_digest(bases_dir, spec)
    older = p.base_version(s) != str(spec["version"])
    if not force and (image_label(ref, DIGEST_LABEL) == digest or (older and image_exists(ref))):
        return ref
    if older and not force:
        print(f"{ref} is an older base version and is missing; building it from the current files")
    elif image_exists(ref) and not force:
        print(f"{ref} is stale (Dockerfile, args or common/ changed since it was built); rebuilding")
    args = [a for k, v in sorted((spec.get("args") or {}).items()) for a in ("--build-arg", f"{k}={v}")]
    _run(["docker", "build", "-f", str(bases_dir / spec["dockerfile"]), *args,
          "--label", f"{DIGEST_LABEL}={digest}", "-t", ref, str(bases_dir)])
    return ref


def build(p: Platform, s: Manifest, rebuild_base: bool = False) -> str:
    if not s.is_custom or s.build is None:
        raise SystemExit(f"{s.name} uses the upstream {s.engine} image {p.image(s)}; nothing to build")
    base = build_base(p, s, force=rebuild_base)
    cmd = ["docker", "build", "-t", p.image(s)]
    if base:
        cmd += ["--build-arg", f"BASE_IMAGE={base}"]
    _run([*cmd, str(p.root / s.build.context)])
    return p.image(s)


def push(p: Platform, s: Manifest) -> None:
    ref = build(p, s)
    base = p.base_image(s)
    if base:
        _run(["docker", "push", base])  # lets other machines rebuild services on the same base
    _run(["docker", "push", ref])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "push"])
    ap.add_argument("name")
    ap.add_argument("--rebuild-base", action="store_true", help="rebuild the base image even if it exists")
    args = ap.parse_args(argv)
    try:
        p = load_platform()
        s = p.get(args.name)
    except PlatformError as exc:
        print("validation FAILED:\n  - " + "\n  - ".join(exc.errors), file=sys.stderr)
        return 1
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    try:
        if args.action == "build":
            print(f"built {build(p, s, rebuild_base=args.rebuild_base)}")
        else:
            push(p, s)
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
