"""VRAM budget for a set of services: sum of gpu.util (engines) + gpu.vram_gb / gpu_total_gb (custom).

    python tools/vram_check.py                         # footprint of every service
    python tools/vram_check.py qwen3-8b default-asr    # one set (names or aliases); exit 1 if over util_limit
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import Manifest, Platform, PlatformError, load_platform


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
    total_fraction: float
    total_gb: float
    limit: float
    gpu_total_gb: float

    @property
    def ok(self) -> bool:
        return self.total_fraction <= self.limit + 1e-9


def row(p: Platform, s: Manifest) -> Row:
    frac = s.vram_fraction(p.config.gpu_total_gb)
    source = "cpu" if s.gpu is None else ("util" if s.gpu.util is not None else "vram_gb")
    return Row(s.name, s.kind, s.engine, frac, frac * p.config.gpu_total_gb, source)


def budget(p: Platform, services: list[Manifest]) -> Budget:
    rows = sorted((row(p, s) for s in services), key=lambda r: (-r.fraction, r.name))
    frac_sum = round(sum(r.fraction for r in rows), 6)
    return Budget(rows, frac_sum, frac_sum * p.config.gpu_total_gb, p.config.util_limit, p.config.gpu_total_gb)


def _row_line(r: Row) -> str:
    if r.source == "cpu":
        return f"    {r.name:<26} {'CPU':>6}   {'-':>7}   {r.kind}/{r.engine}"
    return f"    {r.name:<26} {r.fraction:>6.3f}   {r.gb:>5.1f}GB   {r.kind}/{r.engine} ({r.source})"


def format_budget(b: Budget) -> str:
    status = "OK" if b.ok else "OVER BUDGET"
    lines = [f"{len(b.rows)} services: {b.total_fraction:.3f} of {b.gpu_total_gb:g} GB "
             f"({b.total_gb:.1f} GB, limit {b.limit:.2f}) -> {status}"]
    lines += [_row_line(r) for r in b.rows]
    return "\n".join(lines)


def format_footprints(p: Platform) -> str:
    rows = sorted((row(p, s) for s in p.services), key=lambda r: (-r.fraction, r.name))
    head = (f"GPU {p.config.gpu_total_gb:g} GB, util_limit {p.config.util_limit}. "
            "Any set of services can run together if its sum fits; check one with `just vram NAME...`.")
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
    return 0 if b.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
