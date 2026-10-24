"""aicore: one-line access to platform services by name or alias.

    from aicore import client, endpoint, model_name
    c = client("default-chat")                       # openai.OpenAI for `openai` services
    norm = client("default-normalizer")              # TextServiceClient for `openai-inspired`
    vad = client("default-vad")                      # OIPClient for `oip-v2` (needs aicore[oip])

With AICORE_CATALOG set (e.g. http://127.0.0.1:8700), every client calls the Model Catalog's gateway,
which starts stopped models on demand. `aicore.catalog.Catalog` is the catalog's own API (models,
leases, output index, job plans), and `aicore.datasets.DatasetCatalog` is the Dataset Catalog's (datasets,
versions, the audio store and holds).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .endpoints import AicoreError, UnknownServiceError, available, catalog_url, endpoint, endpoints_path, model_name
from .text import ServiceError, TextServiceClient

if TYPE_CHECKING:
    from .oip import OIPClient

__all__ = [
    "AicoreError",
    "OIPClient",
    "ServiceError",
    "TextServiceClient",
    "UnknownServiceError",
    "available",
    "catalog_url",
    "client",
    "endpoint",
    "endpoints_path",
    "model_name",
]


def __getattr__(name: str) -> Any:
    if name == "OIPClient":  # lazy: numpy is an optional extra
        from .oip import OIPClient

        return OIPClient
    raise AttributeError(f"module 'aicore' has no attribute {name!r}")


def client(name: str, *, host: str | None = None, **kwargs: Any) -> Any:
    """Return the right client for the service's contract.

    openai          -> openai.OpenAI(base_url=.../v1); kwargs go to openai.OpenAI
    openai-inspired -> TextServiceClient; kwargs go to its constructor (e.g. http=, timeout=)
    oip-v2          -> OIPClient;         kwargs go to its constructor
    """
    ep = endpoint(name, host)
    contract = ep["contract"]
    if contract == "openai":
        import openai

        kwargs.setdefault("api_key", os.environ.get("AICORE_API_KEY", "not-needed"))  # no auth on the platform
        return openai.OpenAI(base_url=ep["base_url"], **kwargs)
    if contract == "openai-inspired":
        return TextServiceClient(ep["url"], ep["name"], ep["api"], **kwargs)
    if contract == "oip-v2":
        from .oip import OIPClient

        return OIPClient(ep["url"], ep["name"], ep.get("signature"), **kwargs)
    raise AicoreError(f"service {ep['name']!r} has unknown contract {contract!r}")
