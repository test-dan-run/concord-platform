"""Where each service's model came from, and its upstream model card (README + LICENSE), fetched for the catalog.

    just fetch-cards [NAME ...]      # every service when no name is given

The source of a service is read from what the platform already records, in this order:
`SOURCE.txt` in its weights folder (repo, revision), the manifest comment `# Local copy of <repo> @<sha>`,
then the `source` URL in a custom service's `weights.yaml` (Hugging Face or GitHub).

Fetched files go to ${AI_ASSETS}/model-cards/<host>/<org>/<repo>/<revision>/, never into a weights
folder: every file in a weights folder is part of its build key, so adding a README there would make a
new build. A model needs no fetch when its weights folder holds the upstream README.md and the source came
from SOURCE.txt in that folder (`pinned_readme`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools import identity
from tools.schema import Manifest, Platform, PlatformError, load_platform

CARD_FILES = ("README.md",)
LICENSE_RE = re.compile(r"^(LICEN[CS]E|COPYING)(\.(md|txt))?$", re.IGNORECASE)
_HF_TREE = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+?)(?:/(?:tree|resolve|blob)/([^/]+)(?:/(.*))?)?/?$")
_GH_TREE = re.compile(r"^https://github\.com/([^/]+/[^/]+?)(?:/(?:tree|blob)/([^/]+)(?:/(.*))?)?/?$")
_COMMENT = re.compile(r"#\s*Local copy of (\S+) @(\w+)")
TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Source:
    host: str  # huggingface | github
    repo: str  # org/name
    ref: str | None  # what the platform records: a revision, a short sha, a tag, or None (not pinned)
    subpath: str  # folder inside the repo the weights came from ("" = root)
    origin: str  # SOURCE.txt | manifest comment | weights.yaml
    downloaded: str | None = None

    @property
    def pinned(self) -> bool:
        return bool(self.ref) and (self.host != "huggingface" or bool(re.fullmatch(r"[0-9a-f]{7,40}", self.ref)))

    def url(self, revision: str | None = None) -> str:
        rev = revision or self.ref or "main"
        if self.host == "huggingface":
            return f"https://huggingface.co/{self.repo}/tree/{rev}"
        return f"https://github.com/{self.repo}/tree/{rev}" + (f"/{self.subpath}" if self.subpath else "")


def weights_folder(p: Platform, s: Manifest) -> Path | None:
    src = identity.weights_source(p, s)
    if src.kind == "dir":
        return src.path
    if src.kind == "yaml":
        return p.weights_dir(s)
    return None


def _parse_url(url: str, origin: str) -> Source | None:
    url = url.strip()
    if m := _HF_TREE.match(url):
        return Source("huggingface", m.group(1), m.group(2), "", origin)
    if m := _GH_TREE.match(url):
        return Source("github", m.group(1), m.group(2), (m.group(3) or "").strip("/"), origin)
    return None


def source_of(p: Platform, s: Manifest) -> Source | None:
    folder = weights_folder(p, s)
    src = identity.weights_source(p, s)
    if folder is not None and (folder / "SOURCE.txt").is_file():
        fields = dict(line.split(":", 1) for line in (folder / "SOURCE.txt").read_text().splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get("repo"):
            return Source("huggingface", fields["repo"], fields.get("revision") or None, "", "SOURCE.txt",
                          fields.get("downloaded"))
    if m := _COMMENT.search(p.files[s.name].read_text()):
        return Source("huggingface", m.group(1), m.group(2), "", "manifest comment")
    if src.kind == "yaml":
        spec = yaml.safe_load(src.path.read_text()) or {}
        if spec.get("source") and (found := _parse_url(str(spec["source"]), "weights.yaml")):
            return found
    return None


def pinned_readme(folder: Path | None, src: Source | None) -> Path | None:
    """The README.md in the weights folder, when it is the model's card: only when the source came from
    SOURCE.txt in that folder, which records the repo and revision the folder was downloaded from. Otherwise
    None, and the card is a copy fetched by `just fetch-cards`."""
    if folder is not None and (folder / "README.md").is_file() and src is not None and src.origin == "SOURCE.txt":
        return folder / "README.md"
    return None


def cards_root(p: Platform) -> Path:
    return p.config.assets_path() / "model-cards"


def cache_dirs(p: Platform, src: Source) -> list[Path]:
    """Fetched copies of this source's card, newest first, whose recorded ref matches the source's."""
    base = cards_root(p) / src.host / src.repo
    out = []
    for meta in base.glob("*/meta.json"):
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        rev, asked = data.get("revision") or "", data.get("requested_ref")
        if asked == src.ref or (src.ref and rev.startswith(src.ref)) or (src.ref is None and asked is None):
            out.append((data.get("fetched_at") or "", meta.parent))
    return [d for _, d in sorted(out, reverse=True)]


# ----------------------------------------------------------------------------- fetching

def _get(url: str, http) -> bytes | None:
    r = http.get(url, follow_redirects=True, timeout=TIMEOUT_S)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def fetch(p: Platform, src: Source, http) -> Path:
    """Download README and LICENSE at the source's revision; returns the folder written."""
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    files: dict[str, bytes] = {}
    if src.host == "huggingface":
        api = http.get(f"https://huggingface.co/api/models/{src.repo}/revision/{urllib.parse.quote(src.ref or 'main', safe='')}",
                       follow_redirects=True, timeout=TIMEOUT_S)
        api.raise_for_status()
        info = api.json()
        revision = info["sha"]
        names = [x["rfilename"] for x in info.get("siblings", [])]
        for name in [n for n in names if n in CARD_FILES or LICENSE_RE.match(n)]:
            if (data := _get(f"https://huggingface.co/{src.repo}/resolve/{revision}/{name}", http)) is not None:
                files[name] = data
    else:
        revision = src.ref or "HEAD"
        raw = f"https://raw.githubusercontent.com/{src.repo}/{revision}"
        parts = [x for x in src.subpath.split("/") if x]
        for depth in range(len(parts), -1, -1):  # the weights' folder first, then each parent up to the root
            folder = "/".join(parts[:depth])
            if (data := _get(f"{raw}/{folder + '/' if folder else ''}README.md", http)) is not None:
                files["README.md"] = data
                break
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            if (data := _get(f"{raw}/{name}", http)) is not None:
                files[name] = data
                break
    if "README.md" not in files:
        raise FileNotFoundError(f"{src.repo} has no README.md at {revision}")
    dest = cards_root(p) / src.host / src.repo / revision
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dest / name).write_bytes(data)
    (dest / "meta.json").write_text(json.dumps({
        "host": src.host, "repo": src.repo, "requested_ref": src.ref, "revision": revision, "subpath": src.subpath,
        "fetched_at": fetched_at, "files": sorted(files), "readme_from": None if src.host == "huggingface" else "nearest README.md",
    }, indent=2) + "\n")
    return dest


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(prog="just fetch-cards", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="services (default: all)")
    ap.add_argument("--force", action="store_true", help="fetch again even when a copy exists")
    args = ap.parse_args(argv)
    try:
        p = load_platform()
    except PlatformError as exc:
        print("\n".join(exc.errors), file=sys.stderr)
        return 1
    try:
        services = [p.get(n) for n in args.names] if args.names else p.services
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    failed = 0
    with httpx.Client(headers={"User-Agent": "aiplatform-fetch-cards"}) as http:
        for s in services:
            src = source_of(p, s)
            folder = weights_folder(p, s)
            if src is None:
                why = ("uses no model weights; its README is its card" if s.is_custom and folder is None
                       else "no source recorded (add SOURCE.txt with repo and revision to its weights folder)")
                print(f"  -          {s.name}: {why}")
                continue
            if pinned_readme(folder, src) is not None:
                print(f"  on disk    {s.name}: {src.repo} README.md is in its weights folder")
                continue
            if not args.force and cache_dirs(p, src):
                print(f"  ok         {s.name}: {src.repo} @{(src.ref or 'main')[:12]}")
                continue
            try:
                dest = fetch(p, src, http)
            except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
                failed += 1
                print(f"  FAILED     {s.name}: {src.repo}: {type(exc).__name__}: {exc}")
                continue
            note = "" if src.pinned else " (no pinned revision recorded; fetched the current one)"
            print(f"  fetched    {s.name}: {src.repo} @{dest.name[:12]}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
