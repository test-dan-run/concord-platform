"""GPU service leases: which holder (an app's job, e.g. `bench:job-12`) is using which services.

`stack.py up --holder ID` takes a lease on the services it starts; `up` never stops a leased
service. A lease lasts TTL seconds after its last renewal (`stack.py lease --holder ID`), so a
crashed worker cannot hold services forever. Expired leases are ignored and deleted lazily.

Storage: one JSON file per holder in run/leases/ under the platform root,
`{"holder", "services": [...], "expires_at": epoch}`, written atomically (temp file + rename).
Read-modify-write runs under an exclusive flock on run/leases/.lock, because two apps may call
stack.py at the same moment. Leases use wall-clock TTLs, not pids, because apps may run stack.py in
containers that share the platform directory but not a pid namespace.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TTL = 120


@dataclass
class Lease:
    holder: str
    services: list[str] = field(default_factory=list)
    expires_at: float = 0.0

    def live(self, now: float) -> bool:
        return self.expires_at > now


class Leases:
    """The lease files in one directory. Callers hold `locked()` around read-modify-write."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def path(self, holder: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", holder)[:80]
        return self.dir / f"{safe}-{hashlib.sha1(holder.encode()).hexdigest()[:8]}.json"

    def live(self, now: float | None = None) -> dict[str, Lease]:
        """Live leases by holder; deletes expired or unreadable files."""
        now = time.time() if now is None else now
        out = {}
        for f in sorted(self.dir.glob("*.json")) if self.dir.is_dir() else []:
            try:
                d = json.loads(f.read_text())
                lease = Lease(str(d["holder"]), [str(s) for s in d["services"]], float(d["expires_at"]))
            except (OSError, ValueError, KeyError, TypeError):
                lease = None
            if lease is None or not lease.live(now):
                f.unlink(missing_ok=True)
                continue
            out[lease.holder] = lease
        return out

    def holders_by_service(self, now: float | None = None) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for lease in self.live(now).values():
            for s in lease.services:
                out.setdefault(s, []).append(lease.holder)
        return {s: sorted(h) for s, h in out.items()}

    def write(self, lease: Lease) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.path(lease.holder)
        tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"holder": lease.holder, "services": sorted(set(lease.services)),
                                   "expires_at": lease.expires_at}))
        os.replace(tmp, f)

    def add(self, holder: str, services: list[str], ttl: float) -> list[str]:
        """Add services to the holder's lease and renew it. Returns the names that were not held before."""
        current = self.live().get(holder)
        held = set(current.services) if current else set()
        self.write(Lease(holder, sorted(held | set(services)), time.time() + ttl))
        return sorted(set(services) - held)

    def drop(self, holder: str, services: list[str]) -> None:
        """Remove some services from the holder's lease (the whole file when none are left)."""
        current = self.live().get(holder)
        if current is None:
            return
        rest = [s for s in current.services if s not in services]
        if rest:
            self.write(Lease(holder, rest, current.expires_at))
        else:
            self.path(holder).unlink(missing_ok=True)

    def renew(self, holder: str, ttl: float) -> Lease | None:
        current = self.live().get(holder)
        if current is None:
            return None
        current.expires_at = time.time() + ttl
        self.write(current)
        return current

    def release(self, holder: str) -> Lease | None:
        current = self.live().get(holder)
        self.path(holder).unlink(missing_ok=True)
        return current
