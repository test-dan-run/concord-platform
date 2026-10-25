"""A batch profile whose model fails to start stores "could not start: ...", like verification does."""


def test_a_failed_start_is_stored_as_could_not_start(cat, runtime):
    runtime.fail.add("wavlm-sv")
    cat.scheduler.start()
    try:
        cat.profiler.run("wavlm-sv")
    finally:
        cat.scheduler.shutdown()
    row = cat.db.one("SELECT status, error FROM batch_profile WHERE model_id = 'wavlm-sv'")
    assert row["status"] == "failed"
    assert row["error"] == "could not start: wavlm-sv: wavlm-sv failed to start (fake)"
