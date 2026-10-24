"""GPU check for a set of services: one GPU service at a time, and it gets the whole GPU.

    python tools/vram_check.py                         # minimum footprint of every service
    python tools/vram_check.py qwen3-8b silero-vad     # one set (names or aliases); exit 1 if it cannot run

gpu.util (engines) and gpu.vram_gb / gpu_total_gb (custom) are the least memory a service needs to load.
A set can run when it has at most one GPU service and that minimum is within util_limit; CPU services
run next to it. At start the GPU service gets min(util_limit, (free - 1 GiB) / total) of the device,
measured with nvidia-smi right before the start (start_memory), so memory Windows keeps for the display
under WSL2 is left alone.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import Manifest, Platform, PlatformError, load_platform

RESERVE_GIB = 1.0  # left free at start: vLLM fails when the memory it asks for is not free
# Compose variables render.py puts in the GPU settings of every service; stack.py and the catalog set them
# to the measured value on `docker compose up`. Unset, compose falls back to util_limit x gpu_total_gb.
UTIL_VAR, VRAM_VAR = "AIPLATFORM_GPU_UTIL", "AIPLATFORM_VRAM_GB"


@dataclass(frozen=True)
class Row:
    name: str
    kind: str
    engine: str
    fraction: float
    gb: float
    source: str  # "util", "vram_gb" or "cpu"


@dataclass(frozen=True)
class Budget:
    rows: list[Row]
    total_fraction: float  # sum of the minimums (bundle reports show it)
    total_gb: float
    limit: float
    gpu_total_gb: float

    @property
    def gpu(self) -> list[str]:
        return [r.name for r in self.rows if r.source != "cpu"]

    @property
    def too_big(self) -> list[str]:
        return [r.name for r in self.rows if r.fraction > self.limit + 1e-9]

    @property
    def ok(self) -> bool:
        return len(self.gpu) <= 1 and not self.too_big

    def problem(self) -> str | None:
        if len(self.gpu) > 1:
            return (f"{', '.join(sorted(self.gpu))} are {len(self.gpu)} GPU services; only one GPU service runs at a "
                    "time and it gets the whole GPU, so start one GPU service per stage (CPU services may join it)")
        if self.too_big:
            return f"{', '.join(self.too_big)} needs more than util_limit {self.limit} of the GPU"
        return None


def row(p: Platform, s: Manifest) -> Row:
    frac = s.vram_fraction(p.config.gpu_total_gb)
    source = "cpu" if s.gpu is None else ("util" if s.gpu.util is not None else "vram_gb")
    return Row(s.name, s.kind, s.engine, frac, frac * p.config.gpu_total_gb, source)


def budget(p: Platform, services: list[Manifest]) -> Budget:
    rows = sorted((row(p, s) for s in services), key=lambda r: (-r.fraction, r.name))
    frac_sum = round(sum(r.fraction for r in rows), 6)
    return Budget(rows, frac_sum, frac_sum * p.config.gpu_total_gb, p.config.util_limit, p.config.gpu_total_gb)


# ----------------------------------------------------------------------------- memory at start

def measure_gpu() -> tuple[float, float] | None:
    """(total, free) GiB of GPU 0 from nvidia-smi, device-wide (other processes count); None: cannot measure."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits",
                            "--id=0"], capture_output=True, text=True, timeout=10)
        total, free = (float(v) / 1024 for v in r.stdout.strip().split(","))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return total, free


@dataclass(frozen=True)
class StartMemory:
    fraction: float  # of the device: vLLM --gpu-memory-utilization (SGLang --mem-fraction-static)
    gib: float  # custom services: AISVC_VRAM_GB
    measured: bool

    def env(self) -> dict[str, str]:
        return {UTIL_VAR: f"{self.fraction:.3f}", VRAM_VAR: f"{self.gib:.2f}"}


def start_memory(p: Platform, gpu: tuple[float, float] | None) -> StartMemory:
    """What a GPU service starting now gets: min(util_limit, (free - 1 GiB) / total); unmeasured, util_limit."""
    limit = p.config.util_limit
    if gpu is None:
        return StartMemory(limit, limit * p.config.gpu_total_gb, False)
    total, free = gpu
    frac = math.floor(max(0.0, min(limit, (free - RESERVE_GIB) / total)) * 1000) / 1000  # never more than free
    return StartMemory(frac, math.floor(frac * total * 100) / 100, True)


def min_gib(s: Manifest, total_gib: float) -> float:
    """The least GPU memory the service loads in: gpu.util x the device, or gpu.vram_gb."""
    if s.gpu is None:
        return 0.0
    return s.gpu.util * total_gib if s.gpu.util is not None else float(s.gpu.vram_gb or 0)


def start_env(p: Platform, s: Manifest, gpu: tuple[float, float] | None) -> dict[str, str]:
    """Compose variables for `docker compose up` of s. Raises ValueError when the free memory is under its minimum."""
    if s.gpu is None:
        return {}
    mem = start_memory(p, gpu)
    if mem.measured and gpu and mem.gib + 1e-9 < min_gib(s, gpu[0]):
        raise ValueError(f"{s.name} needs at least {min_gib(s, gpu[0]):.1f} GiB of the GPU to load; "
                         f"{gpu[1]:.1f} GiB is free, so it would get {mem.gib:.1f} GiB "
                         f"(free minus {RESERVE_GIB:g} GiB)")
    return mem.env()


# ----------------------------------------------------------------------------- CLI

def _row_line(r: Row) -> str:
    if r.source == "cpu":
        return f"    {r.name:<26} {'CPU':>6}   {'-':>7}   {r.kind}/{r.engine}"
    return f"    {r.name:<26} {r.fraction:>6.3f}   {r.gb:>5.1f}GB   {r.kind}/{r.engine} ({r.source}, minimum to load)"


def format_budget(b: Budget) -> str:
    gpu = [r for r in b.rows if r.source != "cpu"]
    need = f"needs at least {gpu[0].fraction:.3f} of {b.gpu_total_gb:g} GB ({gpu[0].gb:.1f} GB)" if len(gpu) == 1 else \
        "uses no GPU" if not gpu else f"has {len(gpu)} GPU services"
    lines = [f"{len(b.rows)} services: {need}, limit {b.limit:.2f} -> {'OK' if b.ok else 'REFUSED'}"]
    return "\n".join(lines + [_row_line(r) for r in b.rows])


def format_footprints(p: Platform) -> str:
    rows = sorted((row(p, s) for s in p.services), key=lambda r: (-r.fraction, r.name))
    head = (f"GPU {p.config.gpu_total_gb:g} GB, util_limit {p.config.util_limit}. One GPU service runs at a time and "
            "gets the whole GPU; the numbers are the least each needs to load. Check a set with `just vram NAME...`.")
    return "\n".join([head, *(_row_line(r) for r in rows)])


def main(argv: list[str] | None = None, platform: Platform | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="services (names or aliases) to check together (default: list all)")
    args = ap.parse_args(argv)
    try:
        p = platform or load_platform()
    except PlatformError as exc:
        print("validation FAILED:\n  - " + "\n  - ".join(exc.errors), file=sys.stderr)
        return 1
    if not args.names:
        print(format_footprints(p))
        return 0
    try:
        services = p.resolve(args.names)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    b = budget(p, services)
    print(format_budget(b))
    if not b.ok:
        print(f"error: {b.problem()}", file=sys.stderr)
    return 0 if b.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
