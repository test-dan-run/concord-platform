"""Catalog settings, from CATALOG_* environment variables (and AI_HOST for the gateway's upstream, and
DATASET_STORE or AI_DATA for the Dataset Catalog's store)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tools.schema import PLATFORM_DIR

DEFAULT_PORT = 8700


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _dataset_store() -> Path:
    data = Path(_env("AI_DATA", str(Path.home() / "ai-data"))).expanduser()
    return Path(_env("DATASET_STORE", str(data / "datasets"))).expanduser()


@dataclass
class Settings:
    platform_dir: Path = PLATFORM_DIR
    db_path: Path = field(default_factory=lambda: Path(_env("CATALOG_DB", str(PLATFORM_DIR / "run" / "catalog" / "catalog.db"))))
    bind: str = field(default_factory=lambda: _env("CATALOG_BIND", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("CATALOG_PORT", str(DEFAULT_PORT))))
    # Address the gateway uses for model services: localhost -> published ports; {name} -> service name on ai-net.
    upstream_host: str = field(default_factory=lambda: _env("CATALOG_UPSTREAM_HOST", os.environ.get("AI_HOST") or "localhost"))
    public_url: str = field(default_factory=lambda: _env("CATALOG_PUBLIC_URL", f"http://127.0.0.1:{DEFAULT_PORT}"))
    reconcile_s: float = field(default_factory=lambda: float(_env("CATALOG_RECONCILE_S", "3")))
    lease_ttl_s: float = 90.0  # three missed 30 s heartbeats
    start_timeout_s: float = field(default_factory=lambda: float(_env("AI_STACK_TIMEOUT", str(15 * 60))))
    settle_min_s: float = 60.0
    reuse_tau_s: float = 15 * 60.0  # recency decay for the likelihood of reuse
    # A model a lease brought up is stopped this long after its last lease ends, unless something else needs it
    # (a back-to-back lease on the same model keeps it loaded).
    release_linger_s: float = field(default_factory=lambda: float(_env("CATALOG_RELEASE_LINGER_S", "30")))
    # A model the gateway started for calls without a lease is stopped after this long without a call.
    gateway_idle_s: float = field(default_factory=lambda: float(_env("CATALOG_GATEWAY_IDLE_S", "600")))
    # A GPU model starts only when the device has its whole budget free, plus this margin (CUDA context).
    gpu_margin_gib: float = 0.5
    gpu_free_wait_s: float = 30.0  # how long a start waits for stopped models to give their memory back
    gateway_unmeasured_wait_s: float = 600.0  # how long a call waits for a model whose start was never measured
    verify_on_new_build: bool = field(default_factory=lambda: _env("CATALOG_VERIFY_NEW_BUILDS", "1") == "1")
    docker: bool = True  # False in tests: no Docker, a fake runtime instead
    ui_dir: Path = field(default_factory=lambda: Path(_env("CATALOG_UI_DIR", str(PLATFORM_DIR / "catalog" / "ui" / "dist"))))
    # Hosted endpoints' API keys are encrypted under this key, kept outside PLATFORM_DIR so backups and export
    # bundles never contain it. The container reads it as a Compose secret (catalog/compose.yml).
    master_key_file: Path = field(default_factory=lambda: Path(_env("CATALOG_MASTER_KEY_FILE", str(Path.home() / ".config" / "aiplatform" / "catalog.key"))))
    hosted_reverify_s: float = 24 * 3600.0  # a provider can change the model behind the same name
    # Export bundles; keep it out of the dev/.backups tarballs (or outside PLATFORM_DIR): bundles are tens of GB.
    export_dir: Path = field(default_factory=lambda: Path(_env("CATALOG_EXPORT_DIR", str(PLATFORM_DIR / "run" / "catalog" / "exports"))))

    # The Dataset Catalog's store (catalog/datasets): datasets.db, audio/, versions/ and tmp/. Its records can't be
    # rebuilt, so it lives outside PLATFORM_DIR's run/ folder and is backed up as one folder.
    dataset_store: Path = field(default_factory=lambda: _dataset_store())

    # Cold-start estimates before a build has been measured, seconds.
    estimate_engine_s: float = 180.0
    estimate_custom_gpu_s: float = 30.0
    estimate_cpu_s: float = 10.0
