"""aisvc: app factories for custom platform services, one per API contract.

Service code contains only model loading and a handler; routes, validation, /health (503 while
loading) and permissive CORS come from here. No auth by design.
"""

__version__ = "0.1.0"

from ._core import ServiceState, create_base_app  # noqa: E402
from .audio import encode_audio  # noqa: E402
from .inspired_app import ItnOptions, create_openai_inspired_app  # noqa: E402
from .oip_app import OIPError, create_oip_app  # noqa: E402
from .openai_app import OpenAIError, create_openai_app  # noqa: E402

__all__ = [
    "ItnOptions",
    "OIPError",
    "OpenAIError",
    "ServiceState",
    "create_base_app",
    "create_oip_app",
    "create_openai_app",
    "create_openai_inspired_app",
    "encode_audio",
]
