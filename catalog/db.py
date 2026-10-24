"""SQLite storage. One connection shared under a lock: the catalog's write rate is low and every write is small."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS registration (
    model_id TEXT PRIMARY KEY,
    display_name TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    parent_id TEXT,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS build (
    build_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    weights_hash TEXT NOT NULL,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capability (
    build_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    record TEXT,
    report TEXT,
    rtf REAL,
    verified_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS capability_model ON capability (model_id, updated_at);
CREATE TABLE IF NOT EXISTS cold_start (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL,
    build_key TEXT,
    seconds REAL NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cold_start_model ON cold_start (model_id, build_key);
CREATE TABLE IF NOT EXISTS weights_hash (
    path TEXT PRIMARY KEY,
    quick TEXT NOT NULL,
    content TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    holder TEXT NOT NULL,
    models TEXT NOT NULL,
    state TEXT NOT NULL,
    ttl_s REAL NOT NULL,
    created_at REAL NOT NULL,
    granted_at REAL,
    ready_at REAL,
    heartbeat_at REAL NOT NULL,
    ended_at REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS request_state ON request (state);
CREATE TABLE IF NOT EXISTS usage (
    model_id TEXT PRIMARY KEY,
    calls INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    last_used REAL
);
CREATE TABLE IF NOT EXISTS counter (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS output_index (
    build_key TEXT NOT NULL,
    audio_sha256 TEXT NOT NULL,
    span_start REAL NOT NULL,
    span_end REAL NOT NULL,
    params_hash TEXT NOT NULL,
    owner TEXT NOT NULL,
    output_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (build_key, audio_sha256, span_start, span_end, params_hash, owner)
);
CREATE INDEX IF NOT EXISTS output_index_owner ON output_index (owner, output_id);
CREATE TABLE IF NOT EXISTS app (
    name TEXT PRIMARY KEY,
    base_url TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hosted (
    id TEXT PRIMARY KEY,
    contract TEXT NOT NULL,
    api TEXT NOT NULL,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    auth_type TEXT NOT NULL,
    header_name TEXT,
    key_enc BLOB,
    key_hint TEXT,
    timeout_s REAL NOT NULL,
    max_concurrency INTEGER NOT NULL,
    max_upload_mb REAL NOT NULL,
    signature TEXT,
    data_ack_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS export_job (
    id TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    state TEXT NOT NULL,
    phase TEXT,
    phase_index INTEGER NOT NULL DEFAULT 0,
    phase_count INTEGER NOT NULL DEFAULT 0,
    bytes_done INTEGER NOT NULL DEFAULT 0,
    bytes_expected INTEGER NOT NULL DEFAULT 0,
    bytes_written INTEGER,
    error TEXT,
    failed_phase TEXT,
    path TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
"""


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class DB:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    @contextlib.contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def execute(self, sql: str, args: tuple | dict = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, args).rowcount

    def many(self, sql: str, rows: list[tuple]) -> None:
        with self.tx() as c:
            c.executemany(sql, rows)

    def all(self, sql: str, args: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
