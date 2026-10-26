"""Export starts that race: one job runs, the other gets 409, and no job removes another job's bundle."""

import json
import threading
import time

from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)

from catalog.scheduler import SchedulerError


def test_two_starts_at_once_run_one_job(cat, images, runtime, monkeypatch):
    runtime.save_delay_s = 0.5
    ex = cat.exporter
    real_models = ex.models

    def slow_models():  # the size check sits between the 409 check and the INSERT
        out = real_models()
        time.sleep(0.3)
        return out

    monkeypatch.setattr(ex, "models", slow_models)
    barrier, results = threading.Barrier(2), []

    def go():
        barrier.wait(5)
        try:
            results.append(ex.start({"models": PAIR}))
        except SchedulerError as exc:
            results.append(exc)

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    started = [r for r in results if isinstance(r, dict)]
    refused = [r for r in results if isinstance(r, SchedulerError)]
    assert len(started) == 1 and len(refused) == 1 and refused[0].status == 409, results
    ex.wait(10)
    job = ex.get(started[0]["id"])
    assert job["state"] == "done" and job["available"], job


def test_cancelling_a_job_that_is_not_running_leaves_the_running_one_alone(api, cat, images, runtime, settings):
    runtime.save_delay_s = 1.0
    running = api.post("/api/exports", json={"models": PAIR}).json()
    cat.db.execute("INSERT INTO export_job (id, request, state, phase, path, created_at) VALUES (?, ?, 'queued', 'queued', ?, ?)",
                   ("x-orphan", json.dumps({"models": PAIR}), str(settings.export_dir / "orphan"), "2026-10-26T00:00:00Z"))
    assert api.delete("/api/exports/x-orphan").status_code == 204
    assert api.get("/api/exports/x-orphan").status_code == 404  # nothing ran it: removed
    assert finish(cat, api, running["id"])["state"] == "done"


def test_a_job_whose_folder_exists_fails_without_removing_that_folder(api, cat, images, settings, monkeypatch):
    other = settings.export_dir / "catalog-export-someone-else"
    other.mkdir(parents=True)
    (other / "manifest.json").write_text("{}")
    monkeypatch.setattr(cat.exporter, "_bundle_path", lambda: other)
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])
    assert job["state"] == "failed" and job["failed_phase"] == "checking disk space", job
    assert (other / "manifest.json").read_text() == "{}"
