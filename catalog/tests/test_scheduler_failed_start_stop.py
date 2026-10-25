"""A start that fails stops the container it left half started, whatever the runtime: the clean-up stop is a rule,
not a Docker-only step."""

from catalog_helpers import drain, step

M = "whisper-large-v3"


def test_a_start_that_never_gets_healthy_stops_its_container(cat, runtime):
    s = cat.scheduler
    assert not cat.settings.docker
    cat.settings.start_timeout_s = 0.0
    runtime.start_s = 3600.0  # the container runs but never answers its health check
    r = s.request("bench:job-1", [M])
    drain(cat)
    step(cat)
    r = s.get(r.id)
    assert r.state == "failed" and "not healthy after 0s" in r.error
    assert runtime.started == [M] and runtime.stopped == [M] and M not in runtime.running
    assert s.model_view(M)["state"] == "failed"
