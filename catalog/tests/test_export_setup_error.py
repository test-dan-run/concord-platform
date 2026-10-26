"""An error while an export job sets itself up still ends the job, so later exports can start."""

import json

from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)


def test_a_setup_error_fails_the_job_and_the_next_export_runs(api, cat, images, settings):
    ex = cat.exporter
    ex.db.execute("INSERT INTO export_job (id, request, state, phase, path, created_at) VALUES (?, ?, 'queued', 'queued', ?, ?)",
                  ("x-gone", json.dumps({"models": ["removed-model"], "destination": "bundle", "registry": None,
                                         "images": True, "weights": True, "verification": True}),
                   str(settings.export_dir / "b"), "2026-10-26T00:00:00Z"))
    ex._run("x-gone")  # what the worker thread does: a model removed from the platform since the job was queued
    job = ex.get("x-gone")
    assert job["state"] == "failed" and job["failed_phase"] == "checking disk space" and "removed-model" in job["error"], job
    assert finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])["state"] == "done"
