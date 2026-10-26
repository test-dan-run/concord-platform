"""A failed start stops what it left behind; when that stop also fails, the start's own error is what counts."""

import logging

from catalog_helpers import drain, step

from catalog.runtime import StartError


def test_a_failed_start_whose_clean_up_stop_fails_logs_a_warning(cat, runtime, caplog):
    s = cat.scheduler

    def start(p, svc, env=None):
        raise StartError("container exited/-")

    def stop(p, names):
        raise RuntimeError("docker compose stop whisper-large-v3 exited 1: error during connect")

    runtime.start, runtime.stop = start, stop
    r = s.request("bench:job-1", ["whisper-large-v3"])
    with caplog.at_level(logging.WARNING, logger="catalog.scheduler"):
        drain(cat)
        step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "container exited/-" in r.error and "crashed" not in r.error
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert any("after its failed start failed" in rec.getMessage() for rec in caplog.records)
