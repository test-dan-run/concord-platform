"""The Dataset Catalog: versioned datasets and the audio store they share (README "Dataset Catalog").

Everything lives in one folder, Settings.dataset_store (DATASET_STORE, default ${AI_DATA}/datasets):

    datasets.db                   datasets, versions, audio, holds and uses (SCHEMA below)
    audio/<sha256[0:2]>/<sha256>  original audio bytes, no extension (store.py)
    versions/<dataset_id>/<n>/    one folder per version, in the format of contracts/datasets/version.md
    tmp/                          uploads in progress

store.py keeps the audio, versions.py the datasets and versions, refs.py reads references for the UI,
gc.py cleans up audio nothing needs, and routes.py adds the /api routes to the catalog service's app. The version
format itself (its check, its digest and the reading of references) is aicore.dataset_format, which
FakeDatasetCatalog in aicore.datasets uses too.
"""

from __future__ import annotations

from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    language TEXT,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    created_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_n INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS version (
    dataset_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    parent_n INTEGER,
    digest TEXT NOT NULL,
    manifest TEXT NOT NULL,
    counts TEXT NOT NULL,
    label_types TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '[]',
    published_by TEXT,
    published_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, n)
);
CREATE TABLE IF NOT EXISTS version_file (
    dataset_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    duration REAL,
    sample_rate INTEGER,
    PRIMARY KEY (dataset_id, n, path)
);
CREATE INDEX IF NOT EXISTS version_file_sha256 ON version_file (sha256);
CREATE TABLE IF NOT EXISTS audio (
    sha256 TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    ext TEXT NOT NULL DEFAULT '',
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hold (
    app TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (app, sha256)
);
CREATE INDEX IF NOT EXISTS hold_sha256 ON hold (sha256);
CREATE TABLE IF NOT EXISTS use (
    dataset_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    app TEXT NOT NULL,
    ref TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, n, app, ref)
);
"""


class DatasetError(Exception):
    """An error with its HTTP status. `type` and the extra fields go into the error body, e.g. the latest version
    on a stale parent, or the missing sha256s on a publish."""

    def __init__(self, status: int, message: str, type_: str = "dataset_error", **extra: Any):
        super().__init__(message)
        self.status = status
        self.type = type_
        self.extra = extra
