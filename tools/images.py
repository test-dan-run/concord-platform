"""Build and push custom service images (and their shared base).

    python tools/images.py build NAME [--rebuild-base]
    python tools/images.py push NAME
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import Manifest, Platform, PlatformError, load_bases, load_platform


DIGEST_LABEL = "aiplatform.base-digest"


def image_exists(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref], capture_output=True, check=False).returncode == 0


def image_label(ref: str, label: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", ref],
                       capture_output=True, text=True, check=False)
    return (json.loads(r.stdout or "null") or {}).get(label) if r.returncode == 0 else None


def _ignore_rules(context: Path) -> list[tuple[bool, re.Pattern]]:
    """The context's .dockerignore as (re-include, regex) pairs: ** spans folders, * and ? stay in one.
    Patterns are cleaned as Docker cleans them; one this cannot translate the way Docker does is refused."""
    f = context / ".dockerignore"
    rules = []
    for line in f.read_text().splitlines() if f.is_file() else []:
        pat = line.strip()
        if pat and not pat.startswith("#"):
            keep = pat.startswith("!")
            pat = posixpath.normpath(pat.removeprefix("!").strip()).lstrip("/")
            if "[" in pat or "\\" in pat or any("**" in name and name != "**" for name in pat.split("/")):
                raise ValueError(f".dockerignore: base_digest cannot read {line.strip()!r} the way Docker does; "
                                 "write it without [...], backslashes or ** inside a name, or extend base_digest")
            parts = re.split(r"(\*\*/|\*\*|\*|\?)", pat)
            rx = "".join({"**/": "(.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(s, re.escape(s)) for s in parts)
            rules.append((keep, re.compile(rx)))
    return rules


def _ignored(rel: str, rules: list[tuple[bool, re.Pattern]]) -> bool:
    """Docker's rule: the last pattern that matches the path or one of its parent folders decides."""
    paths = [rel.rsplit("/", n)[0] for n in range(rel.count("/") + 1)]
    out = False
    for keep, rx in rules:
        if any(rx.fullmatch(p) for p in paths):
            out = not keep
    return out


def _instructions(text: str) -> list[list[str]]:
    """The Dockerfile's instructions, each as its stripped lines: as in Docker, a line that ends in a backslash
    goes on in the next one, and blank and comment lines are left out."""
    found, lines = [], []
    for line in map(str.strip, text.splitlines()):
        if line and not line.startswith("#"):
            lines.append(line)
            if not line.endswith("\\"):
                found.append(lines)
                lines = []
    return found + [lines] if lines else found


def _binds(flag: str) -> bool:
    """True for a RUN --mount that can read the build context: type=bind, which is also the default type."""
    if not flag.startswith("--mount="):
        return False
    kind = next((o for o in flag.removeprefix("--mount=").split(",") if o.startswith("type=")), "type=bind")
    return kind == "type=bind"


def _plain(src: str) -> bool:
    """A COPY/ADD source that names files in the context the way it reads: no wildcards, not absolute, no .."""
    return not (any(c in src for c in "*?[") or src.startswith("/") or ".." in src.split("/"))


def base_digest(bases_dir: Path, spec: dict) -> str:
    """Hash of what a base image is built from: its Dockerfile, build args, the .dockerignore and the
    context files its COPY lines read, less what .dockerignore leaves out. Other bases' files do not count.
    A COPY/ADD, RUN --mount or .dockerignore pattern it cannot read exactly is a ValueError: never guess."""
    h = hashlib.sha256(json.dumps({"dockerfile": spec["dockerfile"], "args": spec.get("args") or {}}, sort_keys=True).encode())
    dockerfile = bases_dir / spec["dockerfile"]
    rules = _ignore_rules(bases_dir)
    files = {dockerfile, bases_dir / ".dockerignore"}
    for lines in _instructions(dockerfile.read_text()):
        words = " ".join(ln.removesuffix("\\") for ln in lines).split()
        op, text = words[0].upper(), " ".join(lines)  # Docker reads instructions in any case
        flags = list(itertools.takewhile(lambda w: w.startswith("--"), words[1:]))
        if op == "RUN" and any(_binds(f) for f in flags):
            raise ValueError(f"{dockerfile.name}: base_digest does not read the files a RUN --mount of type bind uses; "
                             f"COPY them instead or extend base_digest: {text}")
        if op in ("COPY", "ADD"):
            srcs = words[1 + len(flags):-1]
            if (lines[0].endswith("\\") or any(f.startswith("--from=") for f in flags)
                    or any(w.startswith("[") for w in words[1:]) or not all(map(_plain, srcs))):
                raise ValueError(f"{dockerfile.name}: base_digest reads one-line COPY/ADD of plain paths in the "
                                 f"context (no wildcards, absolute paths, .., --from or JSON form); write this one "
                                 f"that way or extend base_digest: {text}")  # it would miss files: never guess
            for src in (bases_dir / w for w in srcs):
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
