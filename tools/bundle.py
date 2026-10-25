"""Export services to another machine, as one tarball or a directory.

    python tools/bundle.py NAME [NAME...] [--name BUNDLE] [--out DIR] [--dir [PATH]]
                           [--with-images] [--with-weights] [--platform-tree]
    python tools/bundle.py NAME... | --all  --dry-run [--json]

NAME is a service name or alias. Every bundle has a README with restore steps and `bundle.json`
(services, image refs and ids, sizes, weight paths). Then:

- without --platform-tree: the services' manifests, a standalone compose.yml (images only, no build
  contexts), the endpoints map for aicore and weights.yaml files for custom services;
- --platform-tree: a runnable `platform/` instead: tools, engines, contracts, config, the aicore
  client, custom/_bases and only the chosen services and their custom/<name>/ folders, rendered, so
  `stack.py up NAME` works from inside it without building anything;
- --with-images: images/platform.tar, one `docker save` of every image (shared layers stored once;
  tens of GB with vLLM). An image missing locally fails the bundle (`docker pull` / `just build` first);
- --with-weights: assets/, each service's weights at the same path relative to AI_ASSETS, so
  AI_ASSETS=<bundle>/assets works as-is. Missing or partly downloaded weights fail the bundle.

The tarball is dist/<name>.tar[.gz], named after --name, or after the services when there are three
or fewer. --dir writes a directory instead: <out>/<name>/, or PATH (which must be empty or absent).
--dry-run reports what a bundle of NAMEs would hold (sizes from apparent file sizes, problems) and
writes nothing; --json prints that report as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import render
from tools.identity import PARTIAL
from tools.schema import ASSETS_VAR, BASE_PATHS, MODELS_MOUNT, Manifest, Platform, PlatformError, load_platform
from tools.vram_check import budget

Log = Callable[[str], None]

# What a platform tree holds besides the chosen services. Generated files are left out and rendered fresh.
TREE = ("tools", "engines", "contracts", "clients/aicore", "custom/_bases", "ui", "catalog",
        "config.yaml", "pyproject.toml", "uv.lock", "Justfile", ".env.example", ".gitignore")
TREE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv", "*.egg-info", ".env", "node_modules",
                                     "catalog.html", "endpoints.generated.yaml")


class BundleError(Exception):
    pass


def bundle_name(services: list[Manifest], name: str | None) -> str:
    if name:
        return f"aiplatform-{name}"
    if len(services) <= 3:
        return "aiplatform-" + "-".join(s.name for s in services)
    return f"aiplatform-{len(services)}-services"


def bundle_compose(p: Platform, services: list[Manifest]) -> dict:
    net = p.config.network
    svcs = {s.name: render.compose_service(p, s, p.root, standalone=True) for s in services}
    return {"name": p.config.compose_project, "networks": {net: {"name": net}}, "services": svcs}


# ----------------------------------------------------------------------------- weights

@dataclass
class Weights:
    """Where a service's weights are on this machine and where they go in a bundle's assets/."""

    kind: str  # custom (weights.path) | local (model: /models/X)
    source: Path  # directory on this machine
    path: str  # relative to AI_ASSETS, and to the bundle's assets/
    files: dict[str, Path] = field(default_factory=dict)  # path relative to `path` -> source file
    bytes: int = 0  # apparent size (symlinks followed)
    problems: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": str(self.source), "path": self.path, "files": len(self.files),
                "bytes": self.bytes, "problems": self.problems}


def collect(root: Path, prefix: str = "") -> tuple[dict[str, Path], int, list[str]]:
    """Files under root (symlinks followed), their apparent total size, and broken links."""
    files: dict[str, Path] = {}
    size = 0
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(PARTIAL):
                continue
            f = Path(dirpath) / name
            rel = prefix + f.relative_to(root).as_posix()
            if not f.exists():
                problems.append(f"broken link {rel}")
                continue
            files[rel] = f
            size += f.stat().st_size
    return files, size, problems


def check_index(files: dict[str, Path]) -> list[str]:
    """Every shard named in a *.safetensors.index.json must be there (a partly downloaded model)."""
    problems = []
    for rel, f in files.items():
        if not rel.endswith(".safetensors.index.json"):
            continue
        base = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        try:
            shards = set(json.loads(f.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError) as exc:
            problems.append(f"cannot read {rel}: {exc}")
            continue
        missing = sorted(s for s in shards if base + s not in files)
        if missing:
            problems.append(f"incomplete download: {', '.join(missing)} named in {rel} missing")
    return problems


def weights_of(p: Platform, s: Manifest) -> Weights | None:
    """The service's weights on this machine, or None when it has none (a utility, a weightless custom image)."""
    assets = p.config.assets_path()
    if s.is_custom:
        if s.weights is None:
            return None
        if not s.weights.path.startswith(ASSETS_VAR + "/"):
            return Weights("custom", Path(s.weights.path), s.weights.path,
                           problems=[f"weights.path {s.weights.path} is not under {ASSETS_VAR}"])
        rel = s.weights.path.removeprefix(ASSETS_VAR + "/").strip("/")
        w = Weights("custom", assets / rel, rel)
        missing = f"weights not found at {w.source} (`just fetch-weights {s.name}`)"
    elif s.model:  # the schema allows only /models/... for engine models
        rel = s.model.removeprefix(MODELS_MOUNT + "/").strip("/")
        w = Weights("local", assets / rel, rel)
        missing = f"local model {s.model} not found at {w.source}"
    else:
        return None
    if not w.source.is_dir():
        w.problems.append(missing)
        return w
    w.files, w.bytes, w.problems = collect(w.source)
    if not w.files:
        w.problems.append(f"{w.source} is empty")
    w.problems += check_index(w.files)
    return w


def copy_weights(weights: list[Weights], assets: Path, log: Log) -> None:
    total = sum(w.bytes for w in weights) or 1
    done = 0
    for w in weights:
        log(f"  {w.path} ({w.bytes / 1e6:.1f} MB, {len(w.files)} files)")
        for rel, src in w.files.items():
            dst = assets / w.path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)  # follows symlinks: real files, never links into shared blobs
            size = dst.stat().st_size
            done += size
            if size > 1_000_000:
                log(f"  [{100 * done / total:3.0f}%] {w.path}/{rel}")
    log(f"  [100%] weights copied ({done / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- report

def docker_available() -> bool:
    return shutil.which("docker") is not None


def save_images(refs: list[str], dest: Path) -> None:
    subprocess.run(["docker", "save", "-o", str(dest), *refs], check=True)


def image_facts(ref: str) -> dict[str, Any] | None:
    """{id, bytes} of a local image, or None when it is not present."""
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", ref],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return None
    image_id, size = r.stdout.split()
    return {"id": image_id, "bytes": int(size)}


def report(p: Platform, services: list[Manifest]) -> dict[str, Any]:
    """What a bundle of these services holds: images, weights, sizes and every problem found.

    A row's `image_problem` fails a bundle with images, its `weights.problems` one with weights;
    `warnings` never fail anything."""
    has_docker = docker_available()
    facts = {ref: image_facts(ref) if has_docker else None for ref in sorted({p.image(s) for s in services})}
    rows, problems = [], []
    for s in services:
        ref = p.image(s)
        w = weights_of(p, s)
        image_problem = None
        if has_docker and facts[ref] is None:
            fix = f"`just build {s.name}`" if s.is_custom else f"`docker pull {ref}`"
            image_problem = f"image {ref} is not present locally ({fix} first)"
        row_problems = [f"{s.name}: {x}" for x in [image_problem, *(w.problems if w else [])] if x]
        problems += row_problems
        rows.append({
            "name": s.name, "alias": s.alias, "kind": s.kind, "engine": s.engine, "contract": s.contract,
            "api": s.api, "tasks": list(s.tasks), "description": s.description, "port": s.port,
            "gpu": s.gpu is not None, "gpu_fraction": round(s.vram_fraction(p.config.gpu_total_gb), 4),
            "image": ref, "image_present": None if not has_docker else facts[ref] is not None,
            "image_id": (facts[ref] or {}).get("id"), "image_bytes": (facts[ref] or {}).get("bytes"),
            "image_problem": image_problem,
            "weights": w.to_json() if w else None, "weights_bytes": w.bytes if w else 0,
            "problems": row_problems,
        })
    b = budget(p, services)
    warnings = list(p.warnings)
    if not has_docker:
        warnings.append("docker is not installed: image presence and sizes are unknown")
    if len(b.gpu) > 1:
        warnings.append(f"{len(b.gpu)} GPU services: only one GPU service runs at a time and it gets the whole GPU "
                        "(stack.py starts the one a job stage needs)")
    images_bytes = sum(f["bytes"] for f in facts.values() if f)
    weights_bytes = sum(r["weights_bytes"] for r in rows)
    return {
        "platform": {"gpu_total_gb": p.config.gpu_total_gb, "util_limit": p.config.util_limit,
                     "network": p.config.network, "compose_project": p.config.compose_project},
        "services": rows,
        # Each image counted in full: one `docker save` stores shared layers once, so the tar can be smaller.
        "images": {"refs": list(facts), "bytes": images_bytes},
        "weights_bytes": weights_bytes,
        "bytes": images_bytes + weights_bytes,
        "budget": {"total_fraction": round(b.total_fraction, 4), "util_limit": p.config.util_limit, "ok": b.ok},
        "problems": problems,
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------- writing

def write_platform_tree(p: Platform, services: list[Manifest], dest: Path) -> Platform:
    """A runnable copy of the platform holding only these services, validated and rendered."""
    for rel in TREE:
        src = p.root / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=TREE_IGNORE)
        elif src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest / rel)
    (dest / "services").mkdir(parents=True, exist_ok=True)
    for s in services:
        shutil.copyfile(p.files[s.name], dest / "services" / p.files[s.name].name)
        if s.build:
            shutil.copytree(p.root / s.build.context, dest / s.build.context, ignore=TREE_IGNORE)
    try:
        tree = load_platform(dest)
    except PlatformError as exc:
        raise BundleError("the platform tree does not validate:\n  - " + "\n  - ".join(exc.errors)) from None
    render.write_outputs(tree)
    return tree


def readme(p: Platform, services: list[Manifest], title: str, *, with_images: bool, with_weights: bool,
           platform_tree: bool) -> str:
    order = sorted(services, key=lambda s: (s.gpu is None, -s.vram_fraction(p.config.gpu_total_gb), s.name))
    b = budget(p, services)
    fit = ("one GPU service runs at a time and gets the whole GPU; stop it before you start another"
           if len(b.gpu) > 1 else "it gets the whole GPU" if b.gpu else "none of them uses the GPU")
    if with_weights:
        weights = "in `assets/`, so `AI_ASSETS` is this bundle's `assets/` folder."
    else:
        weights = ("`export AI_ASSETS=/path/to/ai-assets`. Engine models (`/models/<dir>`) must exist as "
                   "`$AI_ASSETS/<dir>`; copy them from the source machine. Custom weights: see "
                   "`weights/<name>.weights.yaml` (place files in `$AI_ASSETS/custom-models/<name>/<version>`).")
    images = ("`docker load -i images/platform.tar`" if with_images else
              "pull or push them first (engine images are public; custom ones need `just push NAME` on the source machine)")
    lines = [
        f"# AI platform bundle: `{title}`",
        "",
        f"GPU: the largest service needs at least {max([r.fraction for r in b.rows] + [0]):.3f} of "
        f"{p.config.gpu_total_gb:g} GB to load (limit {p.config.util_limit}); {fit}.",
        "",
        "| service | kind | contract | api | port | image |",
        "|---|---|---|---|---|---|",
        *[f"| {s.name} | {s.kind} | {s.contract} | {s.api} | {s.port} | `{p.image(s)}` |" for s in services],
        "",
        "## Restore",
        "",
        "1. Install Docker with the NVIDIA container toolkit (driver >= 580 for the CUDA 13 images).",
        f"2. Images: {images}.",
        f"3. Weights: {weights}",
    ]
    if platform_tree:
        assets = "<this bundle>/assets" if with_weights else "/path/to/ai-assets"
        lines += [
            f"4. In `platform/`, write `.env` with `AI_ASSETS={assets}` (an absolute path). Services never "
            "download anything at start, so no internet access is needed.",
            "5. From `platform/`, start services with `just up NAME...` (or `uv run python tools/stack.py up NAME...`): "
            "one at a time, GPU first, each health-gated. Nothing is built, because step 2 loaded every image.",
            "6. Apps: `aicore` reads `platform/clients/aicore/src/aicore/endpoints.generated.yaml`.",
        ]
    else:
        lines += [
            "4. Start services one at a time, waiting for `healthy` (`docker compose ps`) before the next. Only one "
            "GPU service runs at a time: `docker compose stop NAME` before you start another one.",
            "",
            "```bash",
            *[f"docker compose up -d {s.name}" for s in order],
            "```",
            "",
            "5. Apps: `export AICORE_ENDPOINTS=$PWD/endpoints.generated.yaml` and `pip install aicore` (from the platform repo).",
        ]
    lines += [
        "",
        "Base URLs: " + ", ".join(f"`{s.name}` http://127.0.0.1:{s.port}{BASE_PATHS[s.contract]}" for s in services) + ".",
        "",
    ]
    return "\n".join(lines)


def write_bundle(p: Platform, services: list[Manifest], target: Path, *, with_images: bool = False,
                 with_weights: bool = False, platform_tree: bool = False, title: str | None = None,
                 log: Log = print, stage_weights: bool = True) -> dict[str, Any]:
    """Write a bundle directory at target (empty or absent). Returns the report saved as bundle.json.
    stage_weights=False leaves the weights out of target/assets (make_bundle adds them to the tarball from
    where they are); bundle.json and the README still list them."""
    title = title or bundle_name(services, None)
    rep = report(p, services)
    blocking = []
    if with_images and not docker_available():
        blocking.append("--with-images needs docker")
    for row in rep["services"]:
        if with_images and row["image_problem"]:
            blocking.append(f"{row['name']}: {row['image_problem']}")
        if with_weights and row["weights"]:
            blocking += [f"{row['name']}: {x}" for x in row["weights"]["problems"]]
    if blocking:
        raise BundleError("cannot bundle:\n  - " + "\n  - ".join(blocking))
    if target.exists() and any(target.iterdir()):
        raise BundleError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)

    if platform_tree:
        log("== platform tree")
        write_platform_tree(p, services, target / "platform")
        log(f"  platform/ with {', '.join(s.name for s in services)}")
    else:
        names = {s.name for s in services}
        endpoints = {"version": 1, "services": {k: v for k, v in render.build_endpoints(p)["services"].items()
                                                if v["name"] in names}}
        files: dict[str, bytes] = {
            "compose.yml": (render.HEADER + render.dump_yaml(bundle_compose(p, services))).encode(),
            "endpoints.generated.yaml": (render.HEADER + render.dump_yaml(endpoints)).encode(),
        }
        for s in services:
            files[f"services/{s.name}.yaml"] = p.files[s.name].read_bytes()
            if s.build and (p.root / s.build.context / "weights.yaml").is_file():
                files[f"weights/{s.name}.weights.yaml"] = (p.root / s.build.context / "weights.yaml").read_bytes()
        for rel, data in files.items():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(data)

    if with_weights and stage_weights:
        log("== weights")
        copy_weights([w for s in services if (w := weights_of(p, s))], target / "assets", log)
    if with_images:
        log("== images")
        refs = rep["images"]["refs"]
        (target / "images").mkdir(exist_ok=True)
        log(f"  docker save {' '.join(refs)} ({rep['images']['bytes'] / 1e9:.1f} GB before shared layers; "
            "this can take a while)")
        save_images(refs, target / "images" / "platform.tar")

    rep = {"bundle": title, "with_images": with_images, "with_weights": with_weights, "platform_tree": platform_tree,
           **rep}
    (target / "README.md").write_text(readme(p, services, title.removeprefix("aiplatform-"), with_images=with_images,
                                             with_weights=with_weights, platform_tree=platform_tree))
    (target / "bundle.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(f"== done: {target}")
    return rep


def _root_owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Every tarball member is owned by root with mode 0644, whoever wrote the file on this machine."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def write_tar(target: Path, mode: str, title: str, files: dict[str, Path], log: Log) -> None:
    """files (path in the bundle -> file on this machine) as the tarball target, under title/, in path order."""
    total = sum(f.stat().st_size for f in files.values()) or 1
    done = 0
    log(f"== tarball ({total / 1e6:.1f} MB)")
    # dereference: a linked weights file goes in as the file it points to, as copy_weights copies it
    with tarfile.open(target, mode, dereference=True) as tar:
        for rel in sorted(files, key=PurePosixPath):
            tar.add(files[rel], arcname=f"{title}/{rel}", recursive=False, filter=_root_owned)
            size = files[rel].stat().st_size
            done += size
            if size > 1_000_000:
                log(f"  [{100 * done / total:3.0f}%] {rel}")


def make_bundle(p: Platform, services: list[Manifest], out_dir: Path, with_images: bool, name: str | None = None, *,
                with_weights: bool = False, platform_tree: bool = False, log: Log = print) -> Path:
    """The bundle as one tarball in out_dir (uncompressed when it holds images or weights). Only the generated
    files and the docker save are staged in a temporary folder: the weights go into the tarball from where
    they are, so they are not written to disk twice."""
    title = bundle_name(services, name)
    big = with_images or with_weights
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (f"{title}.tar" if big else f"{title}.tar.gz")
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        root = Path(tmp) / title
        write_bundle(p, services, root, with_images=with_images, with_weights=with_weights,
                     platform_tree=platform_tree, title=title, log=log, stage_weights=False)
        files = {f.relative_to(root).as_posix(): f for f in root.rglob("*") if f.is_file()}
        if with_weights:
            files |= {f"assets/{w.path}/{rel}": src for s in services if (w := weights_of(p, s))
                      for rel, src in w.files.items()}
        write_tar(target, "w" if big else "w:gz", title, files, log)
    return target


def format_report(rep: dict[str, Any]) -> str:
    lines = []
    for r in rep["services"]:
        img = "?" if r["image_bytes"] is None else f"{r['image_bytes'] / 1e9:.2f} GB"
        lines.append(f"{r['name']:<36} image {img:>9}  weights {r['weights_bytes'] / 1e9:6.2f} GB  {r['image']}")
    lines.append(f"total: images {rep['images']['bytes'] / 1e9:.2f} GB (before shared layers), "
                 f"weights {rep['weights_bytes'] / 1e9:.2f} GB; VRAM {rep['budget']['total_fraction']:.3f} "
                 f"(limit {rep['budget']['util_limit']})")
    lines += [f"problem: {x}" for x in rep["problems"]] + [f"warning: {x}" for x in rep["warnings"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="services to bundle (names or aliases)")
    ap.add_argument("--all", action="store_true", help="every service (with --dry-run: sizes for the whole registry)")
    ap.add_argument("--name", help="bundle name (default: derived from the services)")
    ap.add_argument("--with-images", action="store_true", help="include one `docker save` of all images")
    ap.add_argument("--with-weights", action="store_true", help="include the weights in assets/ (AI_ASSETS layout)")
    ap.add_argument("--platform-tree", action="store_true", help="include a runnable platform/ with only these services")
    ap.add_argument("--dir", nargs="?", const=True, default=None, metavar="PATH",
                    help="write a directory, not a tarball: PATH, or <out>/<name> without one")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: platform/dist)")
    ap.add_argument("--dry-run", action="store_true", help="report sizes and problems; write nothing")
    ap.add_argument("--json", action="store_true", help="with --dry-run: print the report as JSON")
    args = ap.parse_args(argv)
    if args.json and not args.dry_run:
        ap.error("--json needs --dry-run")
    if bool(args.names) == args.all:
        ap.error("give service names or --all")
    try:
        p = load_platform()
    except PlatformError as exc:
        print("validation FAILED:\n  - " + "\n  - ".join(exc.errors), file=sys.stderr)
        return 1
    try:
        services = list(p.services) if args.all else p.resolve(args.names)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    if args.dry_run:
        rep = report(p, services)
        print(json.dumps(rep, indent=1) if args.json else format_report(rep))
        return 0

    def log(line: str) -> None:
        print(line, flush=True)

    try:
        if args.dir is not None:
            title = bundle_name(services, args.name)
            target = Path(args.dir) if isinstance(args.dir, str) else (args.out or p.root / "dist") / title
            write_bundle(p, services, target.resolve(), with_images=args.with_images, with_weights=args.with_weights,
                         platform_tree=args.platform_tree, title=title, log=log)
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        else:
            target = make_bundle(p, services, args.out or p.root / "dist", args.with_images, args.name,
                                 with_weights=args.with_weights, platform_tree=args.platform_tree, log=log)
            size = target.stat().st_size
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: {' '.join(map(str, exc.cmd[:3]))} ... exited {exc.returncode}", file=sys.stderr)
        return 1
    print(f"wrote {target} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
