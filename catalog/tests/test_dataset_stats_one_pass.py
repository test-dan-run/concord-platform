"""GET /api/store/stats reads the audio table once, grouped by what keeps each file, and gives the numbers the four
separate counts gave."""

import random

import pytest
from catalog_helpers import sha
from fastapi.testclient import TestClient

from catalog.app import create_app

GROUPS = ("audio", "in_versions", "held_only", "neither", "partial")


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app) as c:
        c.dc = app.state.datasets
        yield c
    app.state.datasets.db.close()


def four_counts(db) -> dict:
    """The groups as stats() first counted them, one query each."""
    def group(where):
        r = db.one(f"SELECT COUNT(*) AS files, COALESCE(SUM(size), 0) AS bytes FROM audio a WHERE {where}")
        return {"files": r["files"], "bytes": r["bytes"]}

    listed = "EXISTS (SELECT 1 FROM version_file v WHERE v.sha256 = a.sha256)"
    held = "EXISTS (SELECT 1 FROM hold h WHERE h.sha256 = a.sha256)"
    ready = "a.state = 'ready'"
    return {"audio": group(ready), "in_versions": group(f"{ready} AND {listed}"),
            "held_only": group(f"{ready} AND NOT {listed} AND {held}"),
            "neither": group(f"{ready} AND NOT {listed} AND NOT {held}"), "partial": group("a.state = 'partial'")}


def fill(db, seed: int) -> None:
    r = random.Random(seed)
    shas = [sha(f"{seed}-{i}".encode()) for i in range(r.randrange(0, 80))]
    audio = [(s, r.choice([0, 1, 1000, r.randrange(10 ** 9)]), r.choice(["ready", "ready", "partial"])) for s in shas]
    others = [sha(f"gone-{seed}-{i}".encode()) for i in range(5)]  # listed or held, but not in the store
    with db.tx() as c:
        c.executemany("INSERT INTO audio (sha256, size, ext, state, created_at) VALUES (?, ?, '.wav', ?, 'x')", audio)
        c.executemany("INSERT INTO version_file (dataset_id, n, path, sha256) VALUES (?, ?, ?, ?)",
                      [(f"d{r.randrange(3)}", r.randrange(1, 4), f"f{i}.wav", s)
                       for i, s in enumerate(r.choices(shas + others, k=r.randrange(0, 60)))])
        c.executemany("INSERT OR IGNORE INTO hold (app, sha256, created_at) VALUES (?, ?, 'x')",
                      [(r.choice(["concord-label", "concord-bench"]), s)
                       for s in r.choices(shas + others, k=r.randrange(0, 60))])


def test_stats_match_the_four_counts_on_random_stores(api):
    db = api.dc.db
    for seed in range(60):
        with db.tx() as c:
            for t in ("audio", "version_file", "hold"):
                c.execute(f"DELETE FROM {t}")
        fill(db, seed)
        s = api.get("/api/store/stats").json()
        assert {k: s[k] for k in GROUPS} == four_counts(db), seed
        assert all(type(s[k][f]) is int for k in GROUPS for f in ("files", "bytes"))


def test_stats_read_the_audio_table_once(api):
    fill(api.dc.db, 1)
    seen = []
    api.dc.db._conn.set_trace_callback(seen.append)
    try:
        api.dc.store.stats()
    finally:
        api.dc.db._conn.set_trace_callback(None)
    assert len([q for q in seen if "FROM audio" in q]) == 1, seen
