"""Pydantic models and loaders for config.yaml, engines.yaml and service manifests.

`load_platform()` is the single entry point: it validates every file, runs the cross-manifest
checks (unique names/aliases/ports) and returns a `Platform`, or raises `PlatformError` with
every problem found.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

PLATFORM_DIR = Path(__file__).resolve().parents[1]

CUSTOM_ENGINE = "custom"
CUSTOM_INTERNAL_PORT = 8000
CUSTOM_HEALTH_PATH = "/health"
ASSETS_VAR = "${AI_ASSETS}"
MODELS_MOUNT = "/models"  # where engine services see ${AI_ASSETS} (read-only)

Contract = Literal["openai", "oip-v2", "openai-inspired"]
Kind = Literal["model", "utility"]
OIPDatatype = Literal[
    "BOOL",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "INT8", "INT16", "INT32", "INT64",
    "FP16", "FP32", "FP64",
    "BYTES",
]
BASE_PATHS: dict[str, str] = {"openai": "/v1", "openai-inspired": "/v1", "oip-v2": "/v2"}

_KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class PlatformError(Exception):
    """One or more validation errors across platform files."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _kebab(value: str, what: str) -> str:
    if not _KEBAB.match(value):
        raise ValueError(f"{what} must be kebab-case ([a-z0-9-]), got {value!r}")
    return value


# --------------------------------------------------------------------------- config


class Config(_Strict):
    gpu_total_gb: float = Field(gt=0)
    util_limit: float = Field(gt=0, le=1)
    assets_dir: str
    ai_host: str = "localhost"
    registry: str
    network: str = "ai-net"
    # Docker Compose project of the rendered stack. Change it to run a second copy of the platform
    # (for example an offline kit under test) on the same Docker daemon.
    compose_project: str = "aiplatform"

    @field_validator("compose_project")
    @classmethod
    def _project(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", v):
            raise ValueError(f"compose_project must be lowercase letters, digits, '-' or '_', got {v!r}")
        return v

    @field_validator("registry")
    @classmethod
    def _lowercase_registry(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("registry must be lowercase (Docker image references are lowercase)")
        return v.rstrip("/")

    def assets_path(self) -> Path:
        return Path(os.path.expanduser(os.environ.get("AI_ASSETS") or self.assets_dir))

    def host(self) -> str:
        return os.environ.get("AI_HOST") or self.ai_host


# --------------------------------------------------------------------------- engines


class Engine(_Strict):
    image: str
    contract: Contract
    internal_port: int = Field(gt=0, lt=65536)
    health_path: str
    shm_size: str | None = None
    gpu: bool = True
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def _pinned(cls, v: str) -> str:
        tag = v.rsplit("/", 1)[-1]
        if ":" not in tag and "@" not in tag:
            raise ValueError(f"engine image {v!r} must pin a tag")
        if tag.endswith(":latest"):
            raise ValueError(f"engine image {v!r} uses :latest; pin an exact tag")
        return v

    def render_command(self, *, model: str, name: str, util: float | str | None) -> list[str]:
        values = {
            "model": model,
            "name": name,
            "util": "" if util is None else util if isinstance(util, str) else f"{util:g}",
            "internal_port": str(self.internal_port),
        }
        return [part.format_map(values) for part in self.command]


# --------------------------------------------------------------------------- manifests


class TensorSpec(_Strict):
    name: str
    datatype: OIPDatatype
    shape: list[int] = Field(min_length=1)
    example: list[Any] | None = None  # optional sample value used by conformance and the catalog

    @field_validator("shape")
    @classmethod
    def _dims(cls, v: list[int]) -> list[int]:
        if any(d == 0 or d < -1 for d in v):
            raise ValueError("shape dims must be positive or -1 (variable)")
        return v


class Signature(_Strict):
    inputs: list[TensorSpec] = Field(min_length=1)
    outputs: list[TensorSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Signature:
        for side in ("inputs", "outputs"):
            names = [t.name for t in getattr(self, side)]
            if len(names) != len(set(names)):
                raise ValueError(f"signature {side} have duplicate tensor names: {names}")
        return self


class GPU(_Strict):
    util: float | None = Field(default=None, gt=0, le=1)
    vram_gb: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _exactly_one(self) -> GPU:
        if (self.util is None) == (self.vram_gb is None):
            raise ValueError("gpu must set exactly one of util (engine services) or vram_gb (custom services)")
        return self


class BatchHints(_Strict):
    """How the catalog measures the service's batch profile (catalog/batching.py). Not part of the config hash."""
    max_item_s: float | None = Field(default=None, gt=0)  # the longest input the server takes whole (it cuts longer ones)
    lengths_s: list[float] | None = None  # item lengths to measure at, in seconds (default: short, typical, longest)
    max_n: int | None = Field(default=None, ge=1)  # largest batch the sweep sends (1: the server batches inside one item)


class Build(_Strict):
    context: str
    image: str
    base: str | None = None  # a key of custom/_bases/bases.yaml, or "none"; default: py-cuda if gpu else py-cpu
    # One of that base's older_versions in bases.yaml, to stay on it; default: the base's current version.
    # Left out of the config hash (tools/identity.py): the image id already changes when the base does.
    base_version: str | None = None

    @field_validator("base_version", mode="before")
    @classmethod
    def _version_text(cls, v: Any) -> Any:
        return v if v is None or isinstance(v, str) else str(v)  # YAML reads 1.0 as a float

    @field_validator("image")
    @classmethod
    def _image(cls, v: str) -> str:
        if ":" not in v.rsplit("/", 1)[-1]:
            raise ValueError(f"build.image {v!r} must include a version tag")
        if v.endswith(":latest"):
            raise ValueError(f"build.image {v!r} uses :latest; pin a version")
        if v.replace("{registry}", "") != v.replace("{registry}", "").lower():
            raise ValueError(f"build.image {v!r} must be lowercase")
        return v


class Weights(_Strict):
    path: str
    mount: str = "/models"


class Manifest(_Strict):
    name: str
    kind: Kind
    alias: str | None = None
    engine: str
    contract: Contract
    api: str
    contract_reason: str | None = None
    model: str | None = None
    port: int = Field(ge=1024, le=65535)
    gpu: GPU | None = None
    args: list[str] = Field(default_factory=list)
    description: str | None = None
    tasks: list[str] = Field(min_length=1)  # what the model was trained/built for; catalog tags
    build: Build | None = None
    weights: Weights | None = None
    custom_reason: str | None = None
    signature: Signature | None = None
    batch: BatchHints | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _kebab(v, "name")

    @field_validator("alias")
    @classmethod
    def _alias(cls, v: str | None) -> str | None:
        return None if v is None else _kebab(v, "alias")

    @field_validator("tasks")
    @classmethod
    def _tasks(cls, v: list[str]) -> list[str]:
        for t in v:
            _kebab(t, "task")
        if len(v) != len(set(v)):
            raise ValueError("tasks contain duplicates")
        return v

    @model_validator(mode="after")
    def _rules(self, info: ValidationInfo) -> Manifest:
        """The rules across fields. load_platform passes engines.yaml and api_types.yaml in the context;
        without them the checks that need those files are skipped."""
        ctx = info.context or {}
        errors = self._contract_errors(ctx.get("api_types"))
        if self.engine == CUSTOM_ENGINE:
            errors += self._custom_errors()
        else:
            errors += self._engine_errors(ctx.get("engines"), ctx.get("invalid_engines") or set())
        errors += self._kind_errors()
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _contract_errors(self, api_types: dict[str, dict] | None) -> list[str]:
        errors = []
        if self.contract != "openai" and not self.contract_reason:
            errors.append(f"contract_reason is required when contract is {self.contract!r}")
        if self.contract == "oip-v2" and self.signature is None:
            errors.append("signature is required for contract 'oip-v2'")
        if self.contract != "oip-v2" and self.signature is not None:
            errors.append("signature is only allowed for contract 'oip-v2'")
        if api_types is not None:
            allowed = sorted(api_types.get(self.contract, {}))
            if self.api not in allowed:
                errors.append(f"api {self.api!r} is not valid for contract {self.contract!r} (allowed: {', '.join(allowed)})")
        return errors

    def _custom_errors(self) -> list[str]:
        errors = []
        if self.build is None:
            errors.append("engine 'custom' requires build")
        if self.gpu is not None and self.gpu.vram_gb is None:
            errors.append("custom services declare GPU memory with gpu.vram_gb, not gpu.util")
        return errors

    def _engine_errors(self, engines: dict[str, Engine] | None, invalid_engines: set[str]) -> list[str]:
        """An engine service. A utility on an engine gets its error from _kind_errors."""
        errors = []
        if not self.model:
            errors.append(f"engine {self.engine!r} requires model (a /models/... path)")
        elif not self.model.startswith(MODELS_MOUNT + "/"):
            # Two groups only: custom weights (weights.yaml) and local copies in ${AI_ASSETS}. No hub
            # downloads at start, so the files a build runs are always the files on disk.
            errors.append(f"model {self.model!r} must be a local copy, /models/<dir> in ${{AI_ASSETS}} "
                          f"(`hf download {self.model} --revision <sha> --local-dir $AI_ASSETS/<dir>`)")
        for key in ("build", "weights", "custom_reason"):
            if getattr(self, key) is not None:
                errors.append(f"{key} is not allowed with engine {self.engine!r}")
        if engines is not None:
            errors += self._engines_yaml_errors(engines, invalid_engines)
        return errors

    def _engines_yaml_errors(self, engines: dict[str, Engine], invalid_engines: set[str]) -> list[str]:
        """The engine is in engines.yaml and fits it. invalid_engines are in engines.yaml but failed validation."""
        engine = engines.get(self.engine)
        if engine is None and self.engine in invalid_engines:
            return [f"engine {self.engine!r} is invalid in engines.yaml"]
        if engine is None:
            known = ", ".join(sorted([*engines, CUSTOM_ENGINE]))
            return [f"engine {self.engine!r} is not defined in engines.yaml (known: {known})"]
        errors = []
        if self.contract != engine.contract:
            errors.append(f"contract {self.contract!r} does not match engine {self.engine!r} contract {engine.contract!r}")
        if engine.gpu and (self.gpu is None or self.gpu.util is None):
            errors.append(f"engine {self.engine!r} requires gpu.util")
        if not engine.gpu and self.gpu is not None:
            errors.append(f"engine {self.engine!r} runs on CPU; remove gpu")
        return errors

    def _kind_errors(self) -> list[str]:
        errors = []
        if self.kind == "utility":
            if self.engine != CUSTOM_ENGINE:
                errors.append("kind 'utility' requires engine 'custom'")
            if self.custom_reason is not None:
                errors.append("custom_reason is only for custom models; utilities never need one")
            if self.model is not None:
                errors.append("model is not allowed for kind 'utility'")
        if self.kind == "model" and self.engine == CUSTOM_ENGINE and not self.custom_reason:
            errors.append("custom_reason is required for kind 'model' with engine 'custom' (why can no engine serve it?)")
        return errors

    # ---- derived values used by render / vram / client

    @property
    def is_custom(self) -> bool:
        return self.engine == CUSTOM_ENGINE

    @property
    def base(self) -> str:
        if self.build and self.build.base:
            return self.build.base
        return "py-cuda" if self.gpu else "py-cpu"

    def vram_fraction(self, gpu_total_gb: float) -> float:
        if self.gpu is None:
            return 0.0
        if self.gpu.util is not None:
            return self.gpu.util
        return self.gpu.vram_gb / gpu_total_gb  # type: ignore[operator]


# --------------------------------------------------------------------------- platform


@dataclass
class Platform:
    root: Path
    config: Config
    engines: dict[str, Engine]
    api_types: dict[str, dict[str, dict]]
    services: list[Manifest]  # sorted by name
    files: dict[str, Path]  # service name -> manifest path
    warnings: list[str] = field(default_factory=list)

    def get(self, name_or_alias: str) -> Manifest:
        for s in self.services:
            if name_or_alias in (s.name, s.alias):
                return s
        known = ", ".join(sorted([s.name for s in self.services] + [s.alias for s in self.services if s.alias]))
        raise KeyError(f"unknown service {name_or_alias!r}; known: {known}")

    def resolve(self, names: list[str]) -> list[Manifest]:
        """Manifests for names or aliases, in the order given, without duplicates. KeyError on an unknown one."""
        out: dict[str, Manifest] = {}
        for n in names:
            s = self.get(n)
            out.setdefault(s.name, s)
        return list(out.values())

    def internal_port(self, s: Manifest) -> int:
        return CUSTOM_INTERNAL_PORT if s.is_custom else self.engines[s.engine].internal_port

    def health_path(self, s: Manifest) -> str:
        return CUSTOM_HEALTH_PATH if s.is_custom else self.engines[s.engine].health_path

    def image(self, s: Manifest) -> str:
        if s.is_custom:
            assert s.build is not None
            return s.build.image.replace("{registry}", self.config.registry)
        return self.engines[s.engine].image

    def base_image(self, s: Manifest) -> str | None:
        if not s.is_custom or s.base == "none":
            return None
        return f"{self.config.registry}/{s.base}-base:{self.base_version(s)}"

    def base_version(self, s: Manifest) -> str:
        """The base version a custom service builds on: its pinned build.base_version, else the base's current one."""
        assert s.build is not None
        return s.build.base_version or str(load_bases(self.root)[s.base]["version"])

    def weights_dir(self, s: Manifest) -> Path | None:
        """Host directory of a custom service's weights (${AI_ASSETS} resolved)."""
        if s.weights is None:
            return None
        return Path(s.weights.path.replace(ASSETS_VAR, str(self.config.assets_path())))

    def model_dir(self, s: Manifest) -> Path | None:
        """Host directory of a local model (model: /models/<dir>, which is <dir> in ${AI_ASSETS})."""
        if not s.model or not s.model.startswith(MODELS_MOUNT + "/"):
            return None
        return self.config.assets_path() / s.model.removeprefix(MODELS_MOUNT + "/").strip("/")


def load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_bases(root: Path) -> dict[str, dict]:
    """custom/_bases/bases.yaml: base name -> {dockerfile, version, older_versions, args}."""
    return load_yaml(Path(root) / "custom" / "_bases" / "bases.yaml") or {}


def base_versions(spec: dict) -> list[str]:
    """Every buildable version of one base: the current one first, then its older_versions."""
    return [str(v) for v in [spec["version"], *(spec.get("older_versions") or [])]]


def _fmt_validation(path: Path, exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(manifest)"
        msg = err["msg"].removeprefix("Value error, ")
        out.append(f"{path.name}: {loc}: {msg}")
    return out


# Port band (the hundreds of the port) -> what it holds. A port outside its band is a warning.
PORT_BANDS: dict[int, str] = {
    81: "audio models", 82: "chat/VLM models", 83: "embeddings/rerank models", 84: "other models", 85: "utilities",
}


def port_band(api: str, kind: str) -> int:
    """The port band for a service with this api and kind (the register-model skill uses it too)."""
    if kind == "utility":
        return 85
    if api.startswith("audio."):
        return 81
    if api in ("chat.completions", "completions"):
        return 82
    if api == "embeddings":
        return 83
    return 84


def _port_band(s: Manifest) -> tuple[int, str]:
    band = port_band(s.api, s.kind)
    return band, PORT_BANDS[band]


def load_platform(root: Path | None = None, services_dir: Path | None = None) -> Platform:
    """Load and validate everything. Raises PlatformError listing every problem."""
    root = Path(root or PLATFORM_DIR)
    services_dir = Path(services_dir or root / "services")
    errors: list[str] = []

    try:
        config = Config.model_validate(load_yaml(root / "config.yaml"))
    except ValidationError as exc:
        raise PlatformError(_fmt_validation(root / "config.yaml", exc)) from None

    engines: dict[str, Engine] = {}
    invalid_engines: set[str] = set()
    for name, data in (load_yaml(root / "engines" / "engines.yaml") or {}).items():
        if name == CUSTOM_ENGINE:
            errors.append("engines.yaml: 'custom' is reserved and cannot be an engine name")
            continue
        try:
            engines[name] = Engine.model_validate(data)
        except ValidationError as exc:
            invalid_engines.add(name)
            errors.extend(f"engines.yaml: {e}" for e in _fmt_validation(Path(name), exc))  # "<name>: <field>: ..."

    api_types = load_yaml(root / "contracts" / "api_types.yaml")
    ctx = {"engines": engines, "invalid_engines": invalid_engines, "api_types": api_types}

    services: list[Manifest] = []
    files: dict[str, Path] = {}
    paths = sorted(services_dir.glob("*.yaml")) + sorted(services_dir.glob("*.yml"))
    if not paths:
        errors.append(f"no manifests found in {services_dir}")
    for path in paths:
        try:
            data = load_yaml(path)
            if not isinstance(data, dict):
                raise ValueError("manifest must be a YAML mapping")
            m = Manifest.model_validate(data, context=ctx)
        except ValidationError as exc:
            errors.extend(_fmt_validation(path, exc))
            continue
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if path.stem != m.name:
            errors.append(f"{path.name}: file name must match name {m.name!r} ({m.name}.yaml)")
        services.append(m)
        files[m.name] = path

    # Cross-manifest uniqueness: names, aliases (which share one namespace with names), ports.
    seen_ids: dict[str, Path] = {}
    seen_ports: dict[int, Path] = {}
    for s in services:
        path = files[s.name]
        for ident in filter(None, (s.name, s.alias)):
            if ident in seen_ids and seen_ids[ident] != path:
                errors.append(f"duplicate name/alias {ident!r} in {seen_ids[ident].name} and {path.name}")
            seen_ids.setdefault(ident, path)
        if s.port in seen_ports:
            errors.append(f"port {s.port} collision: {seen_ports[s.port].name} and {path.name}")
        seen_ports.setdefault(s.port, path)
        # gpu.util / vram_gb is the least a service needs to load; it runs alone on the GPU (tools/vram_check.py).
        if s.vram_fraction(config.gpu_total_gb) > config.util_limit + 1e-9:
            errors.append(f"{path.name}: needs {s.vram_fraction(config.gpu_total_gb):.3f} of the GPU, "
                          f"over util_limit {config.util_limit} even when running alone")

    bases = load_bases(root)
    warnings: list[str] = []
    for s in services:
        band, label = _port_band(s)
        if s.port // 100 != band:
            warnings.append(f"{files[s.name].name}: port {s.port} is outside {band}xx, the band for {label}")
        if s.is_custom and s.build:
            ctx_dir = root / s.build.context
            if not (ctx_dir / "Dockerfile").is_file():
                errors.append(f"{files[s.name].name}: build.context {s.build.context!r} has no Dockerfile")
            if s.base != "none" and s.base not in bases:
                errors.append(f"{files[s.name].name}: build.base {s.base!r} is not in custom/_bases/bases.yaml "
                              f"(known: {', '.join(sorted(bases))}, or none)")
            elif s.build.base_version is not None:
                known = base_versions(bases[s.base]) if s.base != "none" else []
                if s.build.base_version not in known:
                    errors.append(f"{files[s.name].name}: build.base_version {s.build.base_version!r} is not a version of "
                                  f"base {s.base!r} (known: {', '.join(known) or 'none'})")
            if (ctx_dir / "weights.yaml").is_file() and s.weights is None:
                errors.append(f"{files[s.name].name}: {s.build.context}/weights.yaml exists, so the manifest needs `weights` "
                              "(where `just fetch-weights` puts them and the container mounts them from)")

    if errors:
        raise PlatformError(errors)
    services.sort(key=lambda s: s.name)
    return Platform(root, config, engines, api_types, services, files, warnings)


def main() -> int:
    """`just validate`."""
    try:
        p = load_platform()
    except PlatformError as exc:
        print("validation FAILED:")
        for e in exc.errors:
            print(f"  - {e}")
        return 1
    if str(PLATFORM_DIR) not in sys.path:  # run as a script: the catalog package is beside tools/
        sys.path.insert(0, str(PLATFORM_DIR))
    from catalog.datasets.versions import check_samples  # the Dataset Catalog's version format, contracts/datasets

    samples = check_samples()
    if samples:
        print("validation FAILED:")
        for e in samples:
            print(f"  - contracts/datasets/samples/{e}")
        return 1
    for w in p.warnings:
        print(f"warning: {w}")
    print(f"ok: {len(p.services)} services, {len(p.engines)} engines; dataset version samples ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
