"""Run the stack: wraps `docker compose -f compose/generated/compose.yml`.

    python tools/stack.py up NAME [NAME...] [--no-evict] [--timeout SECONDS] [--holder ID [--ttl SECONDS]]
    python tools/stack.py down [NAME...]
    python tools/stack.py status
    python tools/stack.py logs NAME
    python tools/stack.py lease --holder ID [--ttl SECONDS]
    python tools/stack.py release --holder ID
    python tools/stack.py leases

NAME is a service name or alias. One GPU service runs at a time and it gets the whole GPU: `up`
validates, renders and refuses a request that names two or more GPU services (CPU services may join
the one GPU service). Another GPU service that is running is stopped first (`--no-evict` refuses
instead). CPU-only services are never stopped. Requested services then start one at a time, the GPU
service first, each waiting until healthy. The GPU service gets min(util_limit, (free - 1 GiB) / total)
of the device, measured with nvidia-smi right before its start (tools/vram_check.py start_memory): vLLM
fails to start when the memory it asks for is not free.

Leases let several apps share the GPU (tools/leases.py). `up --holder ID` (e.g. `bench:job-12`)
leases the requested services to ID for --ttl seconds (default 120); `lease --holder ID` renews
them and `release --holder ID` ends them. `up` never stops a leased service: if another GPU service
is leased, a request for a different GPU service refuses, with or without --holder.

When the Model Catalog is running (catalog/README.md), it owns the GPU: `up`, `down`, `lease`,
`release` and `leases` go through its API instead, so its queue and leases stay the only ones.
`up` then waits its turn in the catalog's queue (no --no-evict: the catalog decides what to stop) and
takes a --ttl of at most 3600 seconds, `lease` ignores --ttl (a catalog lease keeps the TTL it was
taken with) and `status` shows the catalog's lease holders.
Only a refused connection means there is no catalog: one that answers with an error or times out stops
the command. AI_STACK_DIRECT=1 skips the catalog.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import images, render
from tools.leases import DEFAULT_TTL, Leases
from tools.schema import PLATFORM_DIR, Manifest, Platform, PlatformError, load_platform
from tools.vram_check import UTIL_VAR, budget, format_budget, measure_gpu, start_env

DEFAULT_TIMEOUT = int(os.environ.get("AI_STACK_TIMEOUT", str(15 * 60)))
GPU_FREE_WAIT_S = 30  # how long a GPU start waits for stopped services to give their memory back
LEASES_DIR = PLATFORM_DIR / "run" / "leases"
CATALOG_MAX_TTL = 3600  # the longest lease TTL the catalog takes (catalog/app.py LeaseIn.ttl_s)


class StackError(Exception):
    pass


@dataclass
class Container:
    id: str
    name: str
    state: str  # running, exited, created, ...
    health: str  # healthy, unhealthy, starting, or "-" without a healthcheck
    gpu: bool


def _env(p: Platform) -> dict[str, str]:
    return os.environ | {"AI_ASSETS": str(p.config.assets_path())}


def compose_file(p: Platform) -> Path:
    return p.root / "compose" / "generated" / "compose.yml"


def compose(p: Platform, *args: str, capture: bool = False, check: bool = True,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", str(compose_file(p)), *args]
    return subprocess.run(cmd, env=_env(p) | (env or {}), check=check, text=True, capture_output=capture)


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise StackError("docker is not installed or not on PATH")
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise StackError(f"docker daemon not reachable: {r.stderr.strip()}")


def prepare() -> Platform:
    """validate -> render."""
    try:
        p = load_platform()
    except PlatformError as exc:
        raise StackError("validation failed:\n  - " + "\n  - ".join(exc.errors)) from None
    for w in p.warnings:
        print(f"warning: {w}")
    render.write_outputs(p)
    return p


def resolve(p: Platform, names: list[str]) -> list[Manifest]:
    try:
        return p.resolve(names)
    except KeyError as exc:
        raise StackError(exc.args[0]) from None


def ensure_network(name: str) -> None:
    if subprocess.run(["docker", "network", "inspect", name], capture_output=True, check=False).returncode != 0:
        print(f"creating docker network {name}")
        subprocess.run(["docker", "network", "create", name], check=True, capture_output=True)


def ensure_assets(p: Platform, services: list[Manifest]) -> None:
    assets = p.config.assets_path()
    for s in services:
        path = p.weights_dir(s)
        if path is not None and not path.is_dir():
            raise StackError(f"{s.name}: weights not found at {path}; run `just fetch-weights {s.name}`")
        path = p.model_dir(s)
        if path is not None and not path.is_dir():
            raise StackError(f"{s.name}: local model {s.model} not found at {path} (AI_ASSETS={assets})")


def _only_gone(stderr: str) -> bool:
    """True when every error line of `docker inspect` is a missing ID (docker still prints the others)."""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    return bool(lines) and all("no such object" in ln.lower() for ln in lines)


def inspect_project(p: Platform) -> list[dict]:
    """Containers of this compose project (any state) that carry a service name, as dicts with id, name,
    state, health ("-" without a healthcheck), gpu, image and labels. Shared by containers() here and the
    catalog's DockerRuntime. A container removed between `docker ps` and `docker inspect` is left out."""
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={p.config.compose_project}"],
                         capture_output=True, text=True, check=True).stdout.split()
    if not ids:
        return []
    r = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True, check=False)
    if r.returncode != 0 and not _only_gone(r.stderr):
        raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
    found = []
    for c in json.loads(r.stdout or "[]"):
        labels = c["Config"]["Labels"] or {}
        name = labels.get("com.docker.compose.service") or labels.get("aiplatform.name")
        if name:
            found.append({"id": c["Id"], "name": name, "state": c["State"]["Status"],
                          "health": (c["State"].get("Health") or {}).get("Status", "-"),
                          "gpu": labels.get("aiplatform.gpu") == "true", "image": c.get("Image"), "labels": labels})
    return found


def containers(p: Platform) -> dict[str, Container]:
    """Containers of this compose project, keyed by service name (any state)."""
    return {c["name"]: Container(c["id"][:12], c["name"], c["state"], c["health"], c["gpu"]) for c in inspect_project(p)}


def _logs_tail(p: Platform, name: str, lines: int = 80) -> None:
    print(f"\n--- last {lines} log lines of {name} ---", flush=True)
    compose(p, "logs", "--no-color", "--tail", str(lines), name, check=False)
    print("---", flush=True)


def wait_healthy(p: Platform, name: str, timeout: int) -> None:
    start = time.monotonic()
    tty = sys.stdout.isatty()
    last = ""
    next_report = 0  # without a terminal, one progress line about every 30 s
    while True:
        elapsed = int(time.monotonic() - start)
        c = containers(p).get(name)
        state, health = (c.state, c.health) if c else ("missing", "-")
        line = f"  {name}: {state}/{health} {elapsed // 60}m{elapsed % 60:02d}s (timeout {timeout // 60}m)"
        if tty:
            print("\r" + line.ljust(len(last)), end="", flush=True)
        elif elapsed >= next_report:
            print(line, flush=True)
            next_report = (elapsed // 30 + 1) * 30
        last = line
        if health == "healthy" or (state == "running" and health == "-"):
            print(("\r" if tty else "") + f"  {name}: healthy after {elapsed}s".ljust(len(last)), flush=True)
            return
        if state in ("exited", "dead", "missing") or health == "unhealthy":
            print()
            _logs_tail(p, name)
            raise StackError(f"{name} failed to start ({state}/{health})")
        if elapsed >= timeout:
            print()
            _logs_tail(p, name)
            raise StackError(f"{name} not healthy after {timeout}s (raise with --timeout)")
        time.sleep(3)


def lease_store() -> Leases:
    return Leases(LEASES_DIR)


def make_room(p: Platform, wanted: list[Manifest], running: dict[str, Container],
              leased: dict[str, list[str]], evict: bool) -> None:
    """Stop the other running GPU services: the wanted GPU service gets the whole GPU. Leased services are never
    stopped, and a request without a GPU service stops nothing."""
    wanted_names = {s.name for s in wanted}
    if not any(s.gpu for s in wanted):
        return
    known = {s.name: s for s in p.services}
    other_gpu = sorted(n for n, c in running.items() if c.gpu and c.state == "running" and n not in wanted_names)
    # Leased GPU services count even when not running yet: their holder may be starting them right now.
    held = sorted(n for n in leased if n not in wanted_names and (n in other_gpu or (n in known and known[n].gpu)))
    if held:
        who = ", ".join(f"{n} (held by {' and '.join(leased[n])})" for n in held)
        raise StackError(f"the GPU is leased to {who}; only one GPU service runs at a time and leased services "
                         "are never stopped, so retry when it is released")
    if other_gpu:
        if not evict:
            raise StackError(f"{', '.join(other_gpu)} is running on the GPU and only one GPU service runs at a time; "
                             f"stop it (`just down {' '.join(other_gpu)}`) or drop --no-evict")
        print(f"stopping GPU services to give the GPU to {', '.join(s.name for s in wanted if s.gpu)}: "
              f"{', '.join(other_gpu)}", flush=True)
        compose(p, "stop", *other_gpu)


def gpu_env(p: Platform, s: Manifest, wait_s: float = GPU_FREE_WAIT_S) -> dict[str, str]:
    """Compose variables giving s the GPU memory measured as free now (stopped services give theirs back a moment
    after they exit, so this waits up to wait_s). Without nvidia-smi: util_limit x gpu_total_gb."""
    if s.gpu is None:
        return {}
    deadline = time.monotonic() + wait_s
    while True:
        gpu = measure_gpu()
        try:
            env = start_env(p, s, gpu)
        except ValueError as exc:
            if time.monotonic() >= deadline:
                raise StackError(str(exc)) from None
            time.sleep(1)
            continue
        if gpu:
            print(f"  {s.name}: {env[UTIL_VAR]} of the GPU ({gpu[1]:.1f} of {gpu[0]:.1f} GiB free)")
        return env


@contextlib.contextmanager
def renewing(store: Leases, holder: str | None, ttl: int):
    """Keep the holder's lease alive while services start: one start can take longer than the TTL."""
    if not holder:
        yield
        return
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(max(ttl / 3, 1)):
            with store.locked():
                store.renew(holder, ttl)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


def up(names: list[str], timeout: int, evict: bool = True, holder: str | None = None, ttl: int = DEFAULT_TTL) -> None:
    require_docker()
    p = prepare()
    wanted = resolve(p, names)
    b = budget(p, wanted)
    print(format_budget(b))
    if not b.ok:
        raise StackError(b.problem())
    ensure_network(p.config.network)
    ensure_assets(p, wanted)

    wanted_names = {s.name for s in wanted}
    store = lease_store()
    new: list[str] = []
    with store.locked():  # two apps' `up` must not decide evictions from the same snapshot
        running = containers(p)
        make_room(p, wanted, running, store.holders_by_service(), evict)
        if holder:
            new = store.add(holder, sorted(wanted_names), ttl)
            print(f"leased to {holder} for {ttl}s: {', '.join(sorted(wanted_names))}")
    keep_cpu = sorted(n for n, c in running.items() if not c.gpu and c.state == "running" and n not in wanted_names)
    if keep_cpu:
        print(f"leaving CPU services running: {', '.join(keep_cpu)}")

    try:
        with renewing(store, holder, ttl):
            for s in wanted:
                if s.is_custom and not images.image_exists(p.image(s)):
                    print(f"image {p.image(s)} not found locally; building")
                    images.build(p, s)

            order = sorted(wanted, key=lambda s: (s.gpu is None, -s.vram_fraction(p.config.gpu_total_gb), s.name))
            for i, s in enumerate(order, 1):
                c = running.get(s.name)
                tag = "GPU" if s.gpu else "CPU"
                if c and c.state == "running" and c.health in ("healthy", "-"):
                    print(f"[{i}/{len(order)}] {s.name} ({tag}) already running and healthy")
                    continue
                print(f"[{i}/{len(order)}] starting {s.name} ({tag}, :{s.port})", flush=True)
                compose(p, "up", "-d", "--no-deps", "--no-build", s.name, capture=True, env=gpu_env(p, s))
                wait_healthy(p, s.name, timeout)
    except BaseException:
        if new:  # leave no lease behind for a request that failed
            with store.locked():
                store.drop(holder, new)
        raise
    print(f"up: {', '.join(s.name for s in order)}. `just status` for ports; `just catalog` for the UI.")


def down(names: list[str]) -> None:
    require_docker()
    p = prepare()
    if not names:
        _warn_leased(None)
        compose(p, "down")
        return
    services = [s.name for s in resolve(p, names)]
    _warn_leased(services)
    print(f"stopping {', '.join(services)}", flush=True)
    compose(p, "rm", "--stop", "--force", *services)


def _leased() -> dict[str, list[str]]:
    store = lease_store()
    with store.locked():
        return store.holders_by_service()


def _warn_leased(services: list[str] | None) -> None:
    """`down` is the owner's override: it stops leased services too, but says so."""
    hit = {n: h for n, h in _leased().items() if services is None or n in services}
    if hit:
        print("warning: stopping leased services: "
              + ", ".join(f"{n} (held by {' and '.join(h)})" for n, h in sorted(hit.items())))


def _table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())


def status(cat=None) -> None:
    """The service table. With the catalog running (`cat`), LEASED BY comes from its leases, not run/leases."""
    require_docker()
    p = prepare()
    cs = containers(p)
    if cat is None:
        leased = _leased()
    else:
        with catalog_errors(cat):
            leased = {m["id"]: list(m["leased_by"]) for m in cat.models(all=True) if m.get("leased_by")}
    rows = [("NAME", "KIND", "CONTRACT", "VRAM", "STATE", "HEALTH", "PORT", "LEASED BY")]
    for s in p.services:
        c = cs.get(s.name)
        vram = f"{s.vram_fraction(p.config.gpu_total_gb) * p.config.gpu_total_gb:.1f}GB" if s.gpu else "CPU"
        rows.append((s.name, s.kind, s.contract, vram, c.state if c else "-", c.health if c else "-",
                     f"127.0.0.1:{s.port}", ",".join(leased.get(s.name, [])) or "-"))
    _table(rows)


def lease(holder: str, ttl: int) -> None:
    store = lease_store()
    with store.locked():
        held = store.renew(holder, ttl)
    if held is None:
        raise StackError(f"no live lease for {holder} (expired or released); "
                         f"take one with `stack.py up NAME... --holder {holder}`")
    print(f"renewed {holder} for {ttl}s: {', '.join(held.services)}")


def release(holder: str) -> None:
    store = lease_store()
    with store.locked():
        held = store.release(holder)
    print(f"released {holder}: {', '.join(held.services)}" if held else f"{holder} held no leases")


def show_leases() -> None:
    store = lease_store()
    with store.locked():
        live = store.live()
    now = time.time()
    rows = sorted((s, h.holder, f"{int(h.expires_at - now)}s") for h in live.values() for s in h.services)
    if not rows:
        print("no leases")
        return
    _table([("SERVICE", "HOLDER", "EXPIRES IN"), *rows])


def logs(name: str) -> None:
    require_docker()
    p = prepare()
    try:
        s = p.get(name)
    except KeyError as exc:
        raise StackError(exc.args[0]) from None
    compose(p, "logs", "-f", "--tail", "200", s.name, check=False)


# ----------------------------------------------------------------------------- catalog mode

def catalog():
    """The running Model Catalog's client, or None when nothing listens at its URL (then this tool drives Docker
    itself). A catalog that is there but fails or times out stops the command: driving Docker then could stop
    a model the catalog leased to an app. The caller closes the client."""
    if os.environ.get("AI_STACK_DIRECT") == "1":
        return None
    import httpx
    from aicore.catalog import Catalog, CatalogError

    c = Catalog(timeout=5.0)
    try:
        c.health()
    except httpx.ConnectError:
        c.close()
        return None
    except (CatalogError, httpx.HTTPError) as exc:
        c.close()
        raise StackError(f"catalog at {c.url} is not answering ({str(exc) or type(exc).__name__}); "
                         "AI_STACK_DIRECT=1 bypasses it") from None
    return c


def _wait_catalog(cat, names: list[str], timeout: int, lease_ids: list[str]) -> None:
    """Poll until every model runs, keeping our requests alive; print queue position changes."""
    from aicore.catalog import CatalogError

    start, last = time.monotonic(), ""
    live = list(lease_ids)
    while True:
        states = {n: cat.model(n) for n in names}
        for lid in list(live):
            try:
                r = cat.heartbeat(lid)
            except CatalogError as exc:
                # a start request ends as "done" once its models are up; heartbeating it then is a 409
                if exc.status == 409 and cat.get_lease(lid)["state"] == "done":
                    live.remove(lid)
                    continue
                raise
            if r["state"] == "failed":
                raise StackError(f"catalog refused {', '.join(r['models'])}: {r['error']}")
        failed = {n: m["state_detail"] for n, m in states.items() if m["state"] == "failed"}
        if failed:
            raise StackError("; ".join(f"{n} failed to start: {d}" for n, d in failed.items()))
        if all(m["state"] == "running" for m in states.values()):
            return
        line = ", ".join(f"{n}: {m['state']}" + (f" ({m['state_detail']})" if m["state_detail"] else "")
                         for n, m in states.items() if m["state"] != "running")
        if line != last:
            print(f"  {line}", flush=True)
            last = line
        if time.monotonic() - start > timeout:
            raise StackError(f"not running after {timeout}s: {line}")
        time.sleep(2)


def _catalog_up(cat, args: argparse.Namespace) -> None:
    if args.holder and args.ttl > CATALOG_MAX_TTL:  # the catalog would refuse the lease with a 422
        raise StackError(f"--ttl is at most {CATALOG_MAX_TTL} seconds with the catalog running; "
                         f"`stack.py lease --holder {args.holder}` renews the lease to keep it longer")
    p = prepare()
    wanted = resolve(p, args.names)
    if not (b := budget(p, wanted)).ok:
        raise StackError(b.problem())
    names = [s.name for s in wanted]
    print(f"catalog at {cat.url}: queueing {', '.join(names)}" + (f" for {args.holder}" if args.holder else ""))
    if args.no_evict:
        print("note: --no-evict is ignored; the catalog stops only unleased, idle, settled models")
    if args.holder:
        lease = cat.request_lease(args.holder, names, ttl_s=args.ttl)
        ids = [lease["id"]]
    else:
        ids = [r["request"]["id"] for n in names if (r := cat.start(n))["request"]]
    _wait_catalog(cat, names, args.timeout, ids)
    print(f"up: {', '.join(names)}" + (f" (leased to {args.holder})" if args.holder else ""))


def _catalog_down(cat, args: argparse.Namespace) -> None:
    names = args.names or [m["id"] for m in cat.models(all=True) if m["state"] in ("running", "starting")]
    for n in names:
        m = cat.model(n)
        if m["leased_by"]:
            print(f"warning: stopping leased service {m['id']} (held by {' and '.join(m['leased_by'])})")
        cat.stop(n, force=True)
    print(f"stopping {', '.join(names) or 'nothing'}")


def _catalog_lease(cat, args: argparse.Namespace) -> None:
    """`lease` renews and `release` ends every lease of the holder."""
    if args.cmd == "lease" and args.ttl is not None:
        print("note: --ttl is ignored; a catalog lease keeps the TTL it was taken with")
    mine = [r for r in cat.leases() if r["holder"] == args.holder]
    if not mine:
        raise StackError(f"no live lease for {args.holder} (expired or released)" if args.cmd == "lease"
                         else f"{args.holder} held no leases")
    for r in mine:
        cat.heartbeat(r["id"]) if args.cmd == "lease" else cat.release(r["id"])
    models = sorted({m for r in mine for m in r["models"]})
    print(f"{'renewed' if args.cmd == 'lease' else 'released'} {args.holder}: {', '.join(models)}")


def _catalog_leases(cat, args: argparse.Namespace) -> None:
    rows = [(m, r["holder"], r["state"] + (f" #{r['position']}" if r["position"] else ""))
            for r in cat.leases() for m in r["models"]]
    if not rows:
        print("no leases")
    else:
        _table([("SERVICE", "HOLDER", "STATE"), *sorted(rows)])


CATALOG_COMMANDS = {"up": _catalog_up, "down": _catalog_down, "lease": _catalog_lease, "release": _catalog_lease,
                    "leases": _catalog_leases}


@contextlib.contextmanager
def catalog_errors(cat):
    """Catalog API and transport errors as StackError, so they print as one error line."""
    import httpx
    from aicore.catalog import CatalogError

    try:
        yield
    except CatalogError as exc:
        raise StackError(f"catalog: {exc.message}") from None
    except httpx.HTTPError as exc:
        raise StackError(f"catalog at {cat.url} is not answering ({str(exc) or type(exc).__name__})") from None


def via_catalog(cat, args: argparse.Namespace) -> int:
    with catalog_errors(cat):
        CATALOG_COMMANDS[args.cmd](cat, args)
    return 0


def _holder(v: str) -> str:
    if not v.strip():
        raise argparse.ArgumentTypeError("holder must not be empty")
    return v.strip()


def _ttl(v: str) -> int:
    if not v.isdigit() or int(v) < 1:
        raise argparse.ArgumentTypeError("ttl must be a whole number of seconds, at least 1")
    return int(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="start services (names or aliases), one at a time")
    u.add_argument("names", nargs="+")
    u.add_argument("--no-evict", action="store_true", help="fail instead of stopping another running GPU service")
    u.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"seconds per service (default {DEFAULT_TIMEOUT})")
    u.add_argument("--holder", type=_holder, help="lease NAMEs to this holder (e.g. bench:job-12): no other `up` stops them")
    u.add_argument("--ttl", type=_ttl, default=DEFAULT_TTL,
                   help=f"lease lifetime in seconds after each renewal (default {DEFAULT_TTL}; "
                        f"at most {CATALOG_MAX_TTL} with the catalog running)")
    d = sub.add_parser("down", help="stop everything, or the named services")
    d.add_argument("names", nargs="*")
    sub.add_parser("status", help="table of services and container state")
    lg = sub.add_parser("logs", help="follow one service's logs")
    lg.add_argument("name")
    le = sub.add_parser("lease", help="renew all of a holder's leases")
    le.add_argument("--holder", type=_holder, required=True)
    le.add_argument("--ttl", type=_ttl, help=f"seconds from now (default {DEFAULT_TTL}); ignored with the catalog running")
    rl = sub.add_parser("release", help="end all of a holder's leases")
    rl.add_argument("--holder", type=_holder, required=True)
    sub.add_parser("leases", help="live leases: service, holder, and seconds until expiry (with the catalog: lease state)")
    args = ap.parse_args(argv)
    try:
        cat = catalog() if args.cmd in CATALOG_COMMANDS else None
        if cat is not None:
            with cat:
                return via_catalog(cat, args)
        if args.cmd == "up":
            up(args.names, args.timeout, evict=not args.no_evict, holder=args.holder, ttl=args.ttl)
        elif args.cmd == "down":
            down(args.names)
        elif args.cmd == "status":
            cat = catalog()
            with cat if cat is not None else contextlib.nullcontext():
                status(cat=cat)
        elif args.cmd == "logs":
            logs(args.name)
        elif args.cmd == "lease":
            lease(args.holder, DEFAULT_TTL if args.ttl is None else args.ttl)
        elif args.cmd == "release":
            release(args.holder)
        elif args.cmd == "leases":
            show_leases()
    except StackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print(f"error: {' '.join(map(str, exc.cmd[:6]))} ... exited {exc.returncode}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
