"""Output index: (build key, audio sha256, span, params hash) -> which app holds the output, and its id.

Apps keep the outputs; the catalog keeps only this index, so an output either app has already
produced is read from that app instead of being computed again.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .db import DB, now_iso

KEY_FIELDS = ("build_key", "audio_sha256", "span_start", "span_end", "params_hash")
# Output ids are kept as text. Only an integer written the way str(int) writes it (at most 19 digits) comes back
# as a number, so forget() with the id lookup() gave finds the entry again ("007" stays "007").
_INT_ID = re.compile(r"0|-?[1-9][0-9]{0,18}")


class IndexKeyError(ValueError):
    pass


def _span(v: float) -> float:
    return -1.0 if v < 0 else round(v, 3)


def norm_key(k: dict[str, Any]) -> tuple:
    try:
        bk, sha, ph = str(k["build_key"]), str(k["audio_sha256"]), str(k["params_hash"])
        start, end = float(k.get("span_start", -1)), float(k.get("span_end", -1))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise IndexKeyError(f"bad index key {k!r}: needs {', '.join(KEY_FIELDS)}") from exc
    if not bk or bk == "None" or not sha or not ph:
        raise IndexKeyError(f"bad index key {k!r}: build_key, audio_sha256 and params_hash must be set")
    if not (math.isfinite(start) and math.isfinite(end)):  # SQLite would keep NaN as NULL
        raise IndexKeyError(f"bad index key {k!r}: span_start and span_end must be finite numbers")
    start, end = _span(start), _span(end)
    if end < 0:
        start = -1.0
    return bk, sha, start, end, ph


def _id(v: str) -> int | str:
    return int(v) if _INT_ID.fullmatch(v) else v


class OutputIndex:
    def __init__(self, db: DB):
        self.db = db

    def apps(self) -> dict[str, str]:
        return {r["name"]: r["base_url"] for r in self.db.all("SELECT name, base_url FROM app")}

    def register_app(self, name: str, base_url: str) -> dict[str, Any]:
        name, base_url = name.strip(), base_url.strip().rstrip("/")
        if not name or not base_url:
            raise IndexKeyError("name and base_url are required")
        self.db.execute("INSERT INTO app (name, base_url, updated_at) VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                        "base_url = excluded.base_url, updated_at = excluded.updated_at", (name, base_url, now_iso()))
        return {"name": name, "base_url": base_url}

    def lookup(self, app: str, keys: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        urls = self.apps()
        out: list[dict[str, Any] | None] = []
        for k in keys:
            rows = self.db.all("SELECT owner, output_id FROM output_index WHERE build_key = ? AND audio_sha256 = ? "
                               "AND span_start = ? AND span_end = ? AND params_hash = ?", norm_key(k))
            rows.sort(key=lambda r: (r["owner"] != app, r["owner"]))  # the asking app's own entry first
            out.append({"owner": rows[0]["owner"], "output_id": _id(rows[0]["output_id"]),
                        "owner_url": urls.get(rows[0]["owner"])} if rows else None)
        return out

    def count_hits(self, keys: list[dict[str, Any]]) -> int:
        return sum(1 for h in self.lookup("", keys) if h)

    def record(self, app: str, entries: list[dict[str, Any]]) -> int:
        if not app:
            raise IndexKeyError("app is required")
        rows = []
        for e in entries:
            if "output_id" not in e:
                raise IndexKeyError(f"entry {e!r} has no output_id")
            rows.append((*norm_key(e), app, str(e["output_id"]), now_iso()))
        self.db.many("INSERT INTO output_index (build_key, audio_sha256, span_start, span_end, params_hash, owner, "
                     "output_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET "
                     "output_id = excluded.output_id, created_at = excluded.created_at", rows)
        return len(rows)

    def forget(self, app: str, output_ids: list[Any]) -> int:
        n = 0
        for oid in output_ids:
            n += self.db.execute("DELETE FROM output_index WHERE owner = ? AND output_id = ?", (app, str(oid)))
        return n

    def stats(self) -> list[dict[str, Any]]:
        return self.db.all("SELECT owner, COUNT(*) AS outputs FROM output_index GROUP BY owner ORDER BY owner")
