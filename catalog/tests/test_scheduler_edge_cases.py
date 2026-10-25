"""Scheduler edge cases: starts and stops that go wrong, and the starter finishing while a tick runs."""

from catalog_helpers import drain, step


def test_a_start_that_crashes_fails_the_lease_and_frees_the_gpu(cat, runtime):
    s = cat.scheduler
    real = runtime.start

    def start(p, svc, env=None):
        if svc.name == "whisper-large-v3":
            raise FileNotFoundError("[Errno 2] No such file or directory: 'docker'")  # not a StartError
        return real(p, svc, env)

    runtime.start = start
    r = s.request("bench:job-1", ["whisper-large-v3"])
    drain(cat)
    step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "start crashed: [Errno 2]" in r.error
    assert not s.runs["whisper-large-v3"].starting and s.model_view("whisper-large-v3")["state"] == "failed"
    other = s.request("bench:job-2", ["bge-m3"])  # another GPU model does not wait for it
    step(cat)
    assert s.get(other.id).state == "active"
