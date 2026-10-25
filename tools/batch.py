"""Batch size recommendations from the model catalog (catalog/batching.py).

    just batch NAME [--vram GB] [--item-s S | --item-tokens T] [--n N]
    just batch-profile NAME [--wait]

`batch` prints the recommended batch for the memory (default: the full budget) and item length, the memory
batch N needs, and the table of lengths against 16, 24 and 32 GB and the full budget. For a custom GPU image it
also prints the limits its next start gets, the override set by hand and what the running service started with.
`batch-profile` asks the catalog to measure the model now; it leases the model and waits its turn in the queue.
"""

from __future__ import annotations

import argparse
import sys
import time

from aicore.catalog import Catalog, CatalogError


def _limits(d: dict, source: dict | None = None, unset: str = "image default") -> str:
    """max_batch and max_len_s in words, with where each comes from; `unset` stands in for None."""
    parts = []
    for key, label, unit in (("max_batch", "max batch", ""), ("max_len_s", "max length", " s")):
        if d.get(key) is None:
            parts.append(f"{label} {unset}")
        else:
            parts.append(f"{label} {d[key]:g}{unit}" + (f" ({source[key]})" if source else ""))
    return ", ".join(parts)


def show_limits(v: dict) -> None:
    """A custom GPU image's limits: the next start's, the override and the running service's."""
    lim = v.get("limits")
    if not lim:
        return  # the catalog sets limits only for custom GPU images (vLLM sizes its own batches)
    print(f"  next start ({lim['vram_gb']:g} GB): {_limits(lim, lim['source'])}")
    o = v.get("override")
    if o:
        print(f"  override: {_limits(o, unset='from the profile')} (set {o['set_at']})")
    run = v.get("running")
    if run:
        if run.get("started_with"):
            print(f"  running: started with {_limits(run['started_with'])}")
        else:
            print("  running: started outside the catalog, which did not set its limits")


def show(v: dict) -> None:
    print(f"{v['model']}  build {v.get('build_key') or '-'}  profile: {v['status']}"
          + (f" ({v['measured_at']})" if v.get("measured_at") else "")
          + (f"  measuring: {v['measuring']}" if v.get("measuring") else ""))
    if v.get("error"):
        print(f"  last measurement failed: {v['error']}")
    prof = v.get("profile")
    if not prof:
        print(f"  no profile yet: run `just batch-profile {v['model']}`")
        show_limits(v)
        return
    if prof["kind"] == "kv":
        print(f"  vLLM KV cache: {prof['kv_tokens']:,} tokens in {prof['kv_gib']} GiB of {prof['given_gib']} GiB given; "
              f"weights and the rest {prof['fixed_gib']} GiB")
    else:
        print(f"  peak GiB = {prof['base']:.3f} + {prof['a']:.5f} x N x L + {prof['b']:.7f} x N x L^2  "
              f"(L in seconds; {len(prof['points'])} points, {prof['method']})")
    if v.get("note"):
        print(f"  no recommendation: {v['note']}")
    for key in ("recommendation", "need"):
        if key in v:
            r = v[key]
            if key == "recommendation":
                print(f"  {r['vram_gb']:g} GB, items of {r['item']:g} {r['unit']}: batch {r['batch']} "
                      f"(peak {r['peak_gib']:.2f} GiB)")
            else:
                print(f"  batch {r['batch']} of {r['item']:g} {r['unit']} needs {r['vram_gb']:.2f} GB")
    t = v["table"]
    head = ["length (" + t["unit"] + ")"] + [f"{b:g} GB" + (" (full)" if b == t["full_gb"] else "") for b in t["budgets_gb"]]
    print("  " + "  ".join(f"{h:>14}" for h in head))
    for row in t["rows"]:
        print("  " + "  ".join(f"{x:>14}" for x in [f"{row['length']:g}", *map(str, row["batch"])]))
    show_limits(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="batch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("show")
    b.add_argument("name")
    b.add_argument("--vram", type=float)
    b.add_argument("--item-s", type=float)
    b.add_argument("--item-tokens", type=int)
    b.add_argument("--n", type=int)
    m = sub.add_parser("profile")
    m.add_argument("name")
    m.add_argument("--wait", action="store_true", help="wait for the measurement and print the result")
    args = ap.parse_args(argv)
    try:
        with Catalog(timeout=30.0) as cat:
            if args.cmd == "show":
                item_s, item_tokens = args.item_s, args.item_tokens
                if item_s is None and item_tokens is None:  # the catalog uses the one in the profile's unit
                    item_s, item_tokens = 20.0, 1024
                show(cat.batch(args.name, vram_gb=args.vram, item_s=item_s, item_tokens=item_tokens, n=args.n))
                return 0
            r = cat.batch_profile(args.name)
            print(f"{args.name}: measurement {r['measuring']}")
            while args.wait:
                time.sleep(5)
                v = cat.batch(args.name)
                if not v.get("measuring"):
                    show(v)
                    return 0 if v["status"] == "measured" else 1
            return 0
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
