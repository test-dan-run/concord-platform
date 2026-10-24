"""Keep a custom GPU service inside its manifest budget (gpu.vram_gb, passed as AISVC_VRAM_GB)."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("aisvc")


def limit_torch_memory(torch: Any, device: int = 0) -> float | None:
    """Cap the PyTorch caching allocator at AISVC_VRAM_GB on `device`. Returns the fraction set."""
    vram_gb = os.environ.get("AISVC_VRAM_GB")
    if not vram_gb or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, float(vram_gb) * 1024**3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    log.info("limited torch to %.3f of GPU %d (%s GB)", fraction, device, vram_gb)
    return fraction
