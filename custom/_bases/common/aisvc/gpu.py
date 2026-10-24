"""Keep a custom GPU service inside its manifest budget (gpu.vram_gb, passed as AISVC_VRAM_GB) and report its use."""

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


def memory_stats(device: int = 0, reset: bool = False) -> dict[str, Any]:
    """GPU memory as torch sees it (peak, reserved, device total and free) plus the current batch limits.

    `reset` clears the peak after reading it. Without torch or CUDA the body says so instead of failing.
    """
    from .batch import limits_stats

    body: dict[str, Any] = {"torch": False, "cuda": False, "vram_gb_limit": os.environ.get("AISVC_VRAM_GB") or None,
                            "batch": limits_stats(), "reset": False}
    try:
        import torch
    except ImportError:
        return body
    body["torch"] = True
    try:
        if not torch.cuda.is_available():
            return body
        free, total = torch.cuda.mem_get_info(device)
        counts = {"peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": torch.cuda.max_memory_reserved(device),
                  "allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
                  "device_total": total, "device_free": free}
    except Exception as exc:  # a broken CUDA setup is reported, not raised
        body["error"] = f"{type(exc).__name__}: {exc}"
        return body
    body.update(cuda=True, device=device, device_name=torch.cuda.get_device_name(device))
    for k, v in counts.items():
        body[f"{k}_bytes"], body[f"{k}_gib"] = int(v), round(v / 1024**3, 3)
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
        body["reset"] = True
    return body
