"""Endpoint map resolution: name/alias -> service entry + URL."""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

GENERATED = Path(__file__).with_name("endpoints.generated.yaml")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
BASE_PATHS = {"openai": "/v1", "openai-inspired": "/v1", "oip-v2": "/v2"}
CATALOG_TTL_S = 60.0
_remote: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}  # (catalog, name) -> (at, entry)
_remote_lock = threading.Lock()


class AicoreError(Exception):
    pass


class UnknownServiceError(AicoreError, KeyError):
    def __str__(self) -> str:  # KeyError would repr() the message
        return self.args[0]


def endpoints_path() -> Path:
    return Path(os.environ["AICORE_ENDPOINTS"]) if os.environ.get("AICORE_ENDPOINTS") else GENERATED


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("services", {})


def load_endpoints() -> dict[str, dict[str, Any]]:
    path = endpoints_path()
    if not path.is_file():
        raise AicoreError(f"endpoint map not found at {path}; run `just render` in the platform repo "
                          "or set AICORE_ENDPOINTS to a copy of endpoints.generated.yaml")
    return _load(str(path), path.stat().st_mtime)


def available() -> list[str]:
    """Every name and alias in the endpoint map."""
    return sorted(load_endpoints())


def resolve_base(entry: dict[str, Any], host: str | None = None) -> str:
    """Root URL (scheme://host:port) for a service.

    AI_HOST unset/localhost -> published host port on that host.
    Anything else -> internal port, for callers on the docker network; `{name}` in AI_HOST is
    replaced by the service name, so AI_HOST={name} gives http://<service>:<internal_port>.
    """
    host = host or os.environ.get("AI_HOST") or "localhost"
    if host in LOCAL_HOSTS:
        return f"http://{host}:{entry['host_port']}"
    return f"http://{host.replace('{name}', entry['name'])}:{entry['internal_port']}"


def catalog_url() -> str | None:
    """The Model Catalog's root URL from AICORE_CATALOG, or None when apps call services directly."""
    url = os.environ.get("AICORE_CATALOG", "").strip().rstrip("/")
    return url or None


def _from_catalog(catalog: str, name: str) -> dict[str, Any] | None:
    """A model the local map does not list (a hosted endpoint), from `<catalog>/api/models/<name>`, cached 60 s."""
    import httpx

    key = (catalog, name)
    with _remote_lock:
        hit = _remote.get(key)
        if hit and time.monotonic() - hit[0] < CATALOG_TTL_S:
            return hit[1]
    try:
        r = httpx.get(f"{catalog}/api/models/{name}", timeout=10.0)
    except httpx.HTTPError as exc:
        raise AicoreError(f"{name!r} is not in the local endpoint map and the catalog at {catalog} did not answer: {exc}") from None
    entry = None
    if r.status_code == 200:
        rec = r.json()
        entry = {"name": rec.get("served_name") or rec["id"], "id": rec["id"], "alias": None, "kind": rec.get("kind"),
                 "contract": rec["contract"], "api": rec["api"], "engine": rec.get("engine"),
                 "base_path": BASE_PATHS.get(rec["contract"], ""), "description": rec.get("description"),
                 "tasks": rec.get("tasks") or [], "hosted": rec.get("hosted") is not None}
        if rec.get("signature"):
            entry["signature"] = rec["signature"]
    elif r.status_code != 404:
        raise AicoreError(f"the catalog at {catalog} answered {r.status_code} for {name!r}: {r.text[:200]}")
    with _remote_lock:
        _remote[key] = (time.monotonic(), entry)
    return entry


def endpoint(name: str, host: str | None = None) -> dict[str, Any]:
    """The endpoint-map entry for a name or alias, plus `url` (root) and `base_url` (root + base path).

    With AICORE_CATALOG set, `url` is the catalog gateway's route for the service (`<catalog>/m/<name>`),
    which starts a stopped model on demand; `via` says which. An explicit `host` always means a direct call.

    A name the local map does not list is looked up in the catalog (hosted endpoints live only there):
    then `name` is the provider's model name (what goes in request bodies), `id` the catalog id and
    `url` `<catalog>/m/<id>`.
    """
    services = load_endpoints()
    if name not in services:
        catalog = None if host else catalog_url()
        remote = _from_catalog(catalog, name) if catalog else None
        if remote is None:
            hint = (" Hosted models are only reachable through the catalog: set AICORE_CATALOG." if not catalog
                    else f" The catalog at {catalog} does not know it either.")
            raise UnknownServiceError(f"unknown service {name!r}; available: {', '.join(sorted(services)) or '(none)'}."
                                      + hint)
        entry = dict(remote)
        entry["url"] = f"{catalog}/m/{entry['id']}"
        entry["via"] = "catalog"
        entry["base_url"] = entry["url"] + entry["base_path"]
        return entry
    entry = dict(services[name])
    catalog = None if host else catalog_url()
    entry["url"] = f"{catalog}/m/{entry['name']}" if catalog else resolve_base(entry, host)
    entry["via"] = "catalog" if catalog else "direct"
    entry["base_url"] = entry["url"] + entry["base_path"]
    return entry


def model_name(name: str) -> str:
    """The served model name to put in request bodies: the service name, or a hosted model's provider name."""
    return endpoint(name)["name"]
