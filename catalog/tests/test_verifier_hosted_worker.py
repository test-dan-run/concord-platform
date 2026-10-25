"""A hosted verification takes no lease, so it does not wait behind a platform model's verification whose lease is
still queued: hosted endpoints have their own worker."""

import time

from catalog_helpers import step
from test_hosted import ASR, api, provider  # noqa: F401  (fixtures)

from catalog import verifier
from catalog.verifier import HOLDER


def test_a_hosted_verification_does_not_wait_behind_a_queued_platform_lease(api, cat, provider):
    user = cat.scheduler.request("bench:long-job", ["wavlm-sv"])  # a user's lease holds the GPU
    step(cat)
    assert cat.scheduler.get(user.id).state == "active"
    cat.verifier.enqueue("pyannote-diarization")  # a GPU model too: its lease queues behind the user's
    assert api.post("/api/hosted", json=ASR).status_code == 201  # queues hosted-asr after it
    cat.verifier.start()
    try:
        deadline = time.monotonic() + 10
        while cat.verifier.status("hosted-asr") is not None:
            assert time.monotonic() < deadline, cat.verifier.pending
            step(cat, 1)
            time.sleep(0.02)
        assert api.get("/api/models/hosted-asr").json()["capabilities"]["status"] == "verified"
        assert cat.verifier.status("pyannote-diarization") == "running"
        assert [r.state for r in cat.scheduler.live_requests() if r.holder == HOLDER] == ["queued"]
    finally:
        cat.verifier.shutdown()
        for t in cat.verifier._threads:
            t.join(10)
    assert not any(t.is_alive() for t in cat.verifier._threads)
    assert cat.verifier.status("pyannote-diarization") is None  # the waiting verification gave up its lease
    assert [r.holder for r in cat.scheduler.live_requests()] == ["bench:long-job"]


def test_the_daily_hosted_reverify_is_queued_while_a_platform_lease_waits(api, cat, provider, monkeypatch):
    monkeypatch.setattr(verifier, "SCAN_S", 0.05)
    cat.verifier.auto = True
    user = cat.scheduler.request("bench:long-job", ["wavlm-sv"])
    step(cat)
    cat.verifier.enqueue("pyannote-diarization")
    assert api.post("/api/hosted", json=ASR).status_code == 201
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 25 * 3600))
    cat.verifier.start()
    try:
        deadline = time.monotonic() + 10
        while cat.verifier.status("hosted-asr") is not None:
            assert time.monotonic() < deadline, cat.verifier.pending
            step(cat, 1)
            time.sleep(0.02)
        cat.db.execute("UPDATE capability SET verified_at = ? WHERE model_id = 'hosted-asr'", (old,))
        while cat.db.one("SELECT verified_at FROM capability WHERE model_id = 'hosted-asr'")["verified_at"] == old:
            assert time.monotonic() < deadline, cat.verifier.pending
            step(cat, 1)
            time.sleep(0.02)
        assert cat.verifier.status("pyannote-diarization") == "running" and cat.scheduler.get(user.id).state == "active"
    finally:
        cat.verifier.shutdown()
        for t in cat.verifier._threads:
            t.join(10)
