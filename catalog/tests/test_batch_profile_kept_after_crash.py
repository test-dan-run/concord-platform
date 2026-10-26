"""A batch profile measured before its model crashed is kept, as it was before Scheduler.hold."""

from catalog import batching
from catalog.batching import Profile


def test_a_measured_profile_is_kept_when_the_model_crashes_after(cat, runtime, monkeypatch):
    sched = cat.scheduler

    def measure(self, p, s, meta, rid, baseline, given_gib):
        with sched.lock:  # the model crashes and cannot start again just as the sweep ends
            sched._end(sched.requests[rid], "failed", 0.0, "wavlm-sv: container exited/-")
        return Profile(kind="audio", base=1.0, a=0.01, b=0.0, max_item_s=20.0, method="torch")

    monkeypatch.setattr(batching.BatchProfiler, "_measure", measure)
    sched.start()
    try:
        cat.profiler.run("wavlm-sv")
    finally:
        sched.shutdown()
    row = cat.db.one("SELECT status, error FROM batch_profile WHERE model_id = 'wavlm-sv'")
    assert row["status"] == "measured" and row["error"] is None, row["error"]
