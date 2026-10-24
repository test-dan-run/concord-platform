"""What a new manifest has to fit around: names, aliases, ports per band, profile VRAM, engines.

    uv run --quiet python <this file> [--api API] [--kind model|utility] [--profile P]

Run from platform/ (needs the platform venv). With --api, prints the band and the next free port
for that API. With --profile, prints that profile's VRAM breakdown and headroom.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[4]  # the skill lives in <platform>/.claude/skills/register-model/scripts
sys.path.insert(0, str(PLATFORM))

from tools.schema import PlatformError, load_platform  # noqa: E402

BANDS = {81: "audio models", 82: "chat/VLM models", 83: "embeddings/rerank models",
         84: "other models (typically oip-v2)", 85: "utilities"}


def band_for(api: str, kind: str) -> int:
    """Mirrors tools/schema.py _port_band."""
    if kind == "utility":
        return 85
    if api.startswith("audio."):
        return 81
    if api in ("chat.completions", "completions"):
        return 82
    if api == "embeddings":
        return 83
    return 84


def next_free(band: int, used: set[int]) -> int:
    for port in range(band * 100 + 1, band * 100 + 100):
        if port not in used:
            return port
    raise SystemExit(f"band {band}xx is full")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", help="manifest api value, e.g. chat.completions, embeddings, infer")
    ap.add_argument("--kind", default="model", choices=["model", "utility"])
    ap.add_argument("--profile", help="show this profile's VRAM breakdown")
    args = ap.parse_args(argv)

    try:
        p = load_platform(PLATFORM)
    except PlatformError as exc:
        print("platform does not validate; fix this first:\n  - " + "\n  - ".join(exc.errors))
        return 1
    total, limit = p.config.gpu_total_gb, p.config.util_limit

    print("engines:")
    for name, e in sorted(p.engines.items()):
        print(f"  {name:<10} {e.image}  (contract {e.contract})")
    print("  custom     build from custom/<name>/")

    print("\nservices (name / alias / api / port / model):")
    for s in p.services:
        src = s.model or (s.build.context if s.build else "")
        print(f"  {s.name:<26} {s.alias or '-':<20} {s.api:<22} {s.port}  {src}")

    used = {s.port for s in p.services}
    print("\nports in use by band:")
    for band, label in BANDS.items():
        taken = sorted(port for port in used if port // 100 == band)
        print(f"  {band}xx {label:<32} used {taken or '-'}  next free {next_free(band, used)}")

    print(f"\nprofiles (VRAM fraction of {total:g} GB, limit {limit:g}):")
    for prof in p.profiles:
        members = p.in_profile(prof)
        frac = sum(s.vram_fraction(total) for s in members)
        print(f"  {prof:<12} {frac:5.3f}  headroom {limit - frac:5.3f} ({(limit - frac) * total:4.1f} GB)  "
              f"{', '.join(s.name for s in members)}")

    if args.profile:
        members = p.in_profile(args.profile)
        if not members:
            print(f"\nprofile {args.profile!r} does not exist yet (headroom {limit:g}, {limit * total:.1f} GB)")
        else:
            print(f"\n{args.profile}:")
            for s in members:
                f = s.vram_fraction(total)
                print(f"  {s.name:<26} {'CPU' if s.gpu is None else f'{f:.3f}  {f * total:.1f} GB'}")

    if args.api:
        band = band_for(args.api, args.kind)
        allowed = {c: sorted(apis) for c, apis in p.api_types.items() if args.api in apis}
        if not allowed:
            print(f"\napi {args.api!r} is not in contracts/api_types.yaml; known: "
                  + "; ".join(f"{c}: {', '.join(sorted(a))}" for c, a in p.api_types.items()))
            return 1
        print(f"\n--api {args.api}: contract {', '.join(allowed)}, band {band}xx ({BANDS[band]}), "
              f"next free port {next_free(band, used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
