"""A failed start stops its container while the starter still owns the run: a tick during that stop does not take
the container, still listed running, for a model that came up."""

from catalog_helpers import drain, step

M = "whisper-large-v3"


def test_a_tick_during_a_failed_starts_clean_up_stop_leaves_the_lease_starting(cat, runtime):
    s = cat.scheduler
    cat.settings.start_timeout_s = 0.0
    runtime.healthy = lambda p, svc, url: False  # listed running and healthy, but the HTTP check never passes
    seen = {}
    real_stop = runtime.stop

    def stop(p, names):
        s.tick()  # docker compose stop is still running: the container is listed running and healthy
        run = s.runs[M]
        seen.update(running=run.running, starting=run.starting, failed=run.failed, state=s.get(rid).state)
        real_stop(p, names)

    runtime.stop = stop
    rid = s.request("bench:job-1", [M]).id
    drain(cat)
    assert seen == {"running": False, "starting": True, "failed": None, "state": "starting"}
    runtime.stop = real_stop
    step(cat)
    r = s.get(rid)
    assert r.state == "failed" and "not healthy after 0s" in r.error
    assert M not in runtime.running and runtime.started == [M] and s.model_view(M)["state"] == "failed"
