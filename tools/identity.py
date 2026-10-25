"""What makes a build: the config hash of a manifest and where its weights come from.

The Model Catalog keys verification results and indexed outputs on a *build key* =
sha256(image id, config hash, weights hash). This module has the parts that need only the manifests,
so `render.py` can stamp the config hash on each container (label `aiplatform.config_hash`) and a
running container reports the config it was started with.

- Config hash: the manifest fields that change outputs. Port, alias, description, tasks and the GPU
  size are left out, so editing them does not make a new build.
- Weights: custom services with weights.yaml are identified by its file list and sha256s; every other
  weights folder (engine models' /models/<dir>, custom "existing local copy" folders) by its content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.schema import Manifest, Platform

IGNORED_FILES = {"SOURCE.txt"}  # provenance notes, not weights
PARTIAL = (".lock", ".incomplete", ".part")  # download leftovers: never weights, never copied (tools/bundle.py)


def _digest(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:n]


def config_fields(p: Platform, s: Manifest) -> dict[str, Any]:
    fields: dict[str, Any] = {"engine": s.engine, "contract": s.contract, "api": s.api}
    if s.signature:
        fields["signature"] = s.signature.model_dump(exclude_none=True)
    if s.is_custom:
        assert s.build is not None
        # The base's version (build.base_version) is left out: the image id changes when the base does.
        fields["build"] = {"context": s.build.context, "base": s.base}
        if s.weights:
            fields["weights"] = s.weights.model_dump()
    else:
        engine = p.engines[s.engine]
        fields.update({"image": engine.image, "command": engine.command, "env": engine.env, "model": s.model,
                       "args": list(s.args)})
    return fields


def config_hash(p: Platform, s: Manifest) -> str:
    return _digest(config_fields(p, s))


@dataclass(frozen=True)
class WeightsSource:
    kind: str  # "yaml" (hash of weights.yaml's files) | "dir" (content hash of a folder) | "none"
    path: Path | None = None


def weights_source(p: Platform, s: Manifest) -> WeightsSource:
    if s.is_custom:
        assert s.build is not None
        spec = p.root / s.build.context / "weights.yaml"
        if spec.is_file():
            return WeightsSource("yaml", spec)
        if s.weights:
            return WeightsSource("dir", p.weights_dir(s))
        return WeightsSource("none")
    model = p.model_dir(s)
    return WeightsSource("dir", model) if model is not None else WeightsSource("none")


def weights_yaml_hash(spec: Path) -> str:
    data = yaml.safe_load(spec.read_text()) or {}
    files = sorted((str(f.get("path")), str(f.get("sha256"))) for f in data.get("files") or [])
    return _digest({"version": data.get("version"), "files": files})


def weight_files(root: Path) -> list[tuple[str, Path]]:
    """(relative path, file) of every weights file under root, sorted; hidden entries, SOURCE.txt and download
    leftovers skipped."""
    out = []
    for f in root.rglob("*"):
        rel = f.relative_to(root)
        if (any(part.startswith(".") for part in rel.parts) or rel.name in IGNORED_FILES or rel.name.endswith(PARTIAL)
                or not f.is_file()):
            continue
        out.append((rel.as_posix(), f))
    return sorted(out)


def quick_fingerprint(root: Path) -> str:
    """Cheap stand-in for the content hash: path, size and mtime of every file. Changes when a file does."""
    rows = []
    for rel, f in weight_files(root):
        st = f.stat()
        rows.append((rel, st.st_size, st.st_mtime_ns))
    return _digest(rows, 32)


def content_hash(root: Path, chunk: int = 8 << 20) -> str:
    """sha256 over every file's path and bytes. Slow for large models; callers cache it by quick_fingerprint."""
    h = hashlib.sha256()
    for rel, f in weight_files(root):
        fh_hash = hashlib.sha256()
        with open(f, "rb") as fh:
            while block := fh.read(chunk):
                fh_hash.update(block)
        h.update(f"{rel}\0{fh_hash.hexdigest()}\n".encode())
    return h.hexdigest()[:16]


def build_key(image_id: str, config: str, weights: str) -> str:
    return hashlib.sha256(f"{image_id}\n{config}\n{weights}".encode()).hexdigest()[:16]
