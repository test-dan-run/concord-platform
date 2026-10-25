"""aicore.catalog: each case runs against FakeCatalog and against Catalog talking to the catalog service in
process, so the fake keeps to the server's rules where apps rely on them (leases, the output index, apps and
plans). The fake stays permissive where the apps' tests rely on that (see the last test)."""

import contextlib
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from aicore.catalog import Catalog, CatalogError, FakeCatalog, Lease, index_key
from catalog.app import create_app

KEY = index_key("b" * 16, "f" * 64, None, None, "p" * 16)


@contextlib.contextmanager
def real(cat) -> Iterator[Catalog]:
    """A Catalog on the catalog service's app, in process."""
    app = create_app(cat, start=False)
    with TestClient(app) as http:
        yield Catalog("http://testserver", http=http)
    app.state.datasets.db.close()


@pytest.fixture(params=["fake", "real"])
def c(request, cat):
    with real(cat) as server:
        yield FakeCatalog(server.models(all=True)) if request.param == "fake" else server


def error(fn, *args) -> CatalogError:
    with pytest.raises(CatalogError) as exc:
        fn(*args)
    return exc.value


# ----------------------------------------------------------------------------- leases

def test_unknown_lease_is_a_404(c):
    for fn in (c.get_lease, c.heartbeat, c.release):
        exc = error(fn, "zzz")
        assert (exc.status, exc.message) == (404, "unknown lease 'zzz'")


def test_heartbeat_of_an_ended_lease_is_a_409_and_the_lease_is_lost(c):
    lid = c.request_lease("x:1", ["silero-vad"])["id"]
    assert c.heartbeat(lid)["state"] in ("queued", "starting", "active")
    c.release(lid)
    exc = error(c.heartbeat, lid)
    assert (exc.status, exc.message) == (409, f"lease {lid} is released")
    lease = Lease(c, c.request_lease("x:2", ["silero-vad"]), heartbeat_s=0.01)
    c.release(lease.id)
    deadline = time.monotonic() + 5
    while lease.lost is None and time.monotonic() < deadline:
        time.sleep(0.01)
    lease.release()
    assert lease.lost == f"lease {lease.id} is released" and not lease.active


def test_lease_resolves_an_alias(c):
    lease = c.request_lease("x:1", ["default-vad"])
    assert lease["models"] == ["silero-vad"] and c.get_lease(lease["id"])["models"] == ["silero-vad"]
    if isinstance(c, FakeCatalog):
        assert c.calls[-1] == ("lease", ("x:1", ("default-vad",)))  # the call log keeps the models as given


# ----------------------------------------------------------------------------- output index and apps

def test_lookup_gives_digit_string_output_ids_as_ints(c):
    c.index_record("bench", [{**KEY, "output_id": "7"}, {**KEY, "params_hash": "q" * 16, "output_id": "-3"},
                             {**KEY, "params_hash": "r" * 16, "output_id": "run-4"}])
    hits = c.index_lookup("label", [KEY, {**KEY, "params_hash": "q" * 16}, {**KEY, "params_hash": "r" * 16}])
    assert [h["output_id"] for h in hits] == [7, -3, "run-4"]


def test_forget_compares_output_ids_as_strings(c):
    c.index_record("bench", [{**KEY, "output_id": 9}, {**KEY, "params_hash": "q" * 16, "output_id": "10"}])
    c.index_record("label", [{**KEY, "params_hash": "r" * 16, "output_id": 9}])
    assert c.index_forget("bench", ["9", 10]) == 2
    assert c.index_lookup("x", [KEY, {**KEY, "params_hash": "q" * 16}]) == [None, None]
    assert c.index_lookup("x", [{**KEY, "params_hash": "r" * 16}])[0]["owner"] == "label"  # another app's entry


def test_register_app_strips_the_name_and_url(c):
    assert c.register_app(" bench ", " http://bench:8000/ ") == {"name": "bench", "base_url": "http://bench:8000"}
    assert c.apps() == [{"name": "bench", "base_url": "http://bench:8000"}]
    c.index_record("bench", [{**KEY, "output_id": 1}])
    assert c.index_lookup("label", [KEY])[0]["owner_url"] == "http://bench:8000"


# ----------------------------------------------------------------------------- plans

def test_plan_steps_have_the_servers_fields(c):
    [step] = c.plan([{"name": "seg", "models": ["silero-vad"], "keys": {"silero-vad": [KEY]}}])["steps"]
    assert sorted(step) == ["cold_start_measured", "cold_start_s", "hits", "items", "load", "missing", "model",
                            "skip", "stage"]
    assert step["cold_start_measured"] is False and (step["model"], step["items"], step["hits"]) == ("silero-vad", 1, 0)


def test_plan_reports_the_resolved_id(c, request):
    if not isinstance(c, FakeCatalog):
        request.applymarker(pytest.mark.xfail(raises=KeyError, strict=False,
                                              reason="the server's planner fails for an alias until lane H's H1"))
    assert [s["model"] for s in c.plan([{"name": "seg", "models": ["default-vad"]}])["steps"]] == ["silero-vad"]


def test_plan_without_stages_is_a_422(c):
    exc = error(c.plan, [])
    assert (exc.status, exc.message) == (422, "stages must not be empty")


# ----------------------------------------------------------------------------- fake only

def test_fake_stays_permissive_where_apps_rely_on_it():
    fake = FakeCatalog([{"id": "silero-vad", "aliases": ["default-vad"], "build_key": "b" * 16, "registered": False}])
    assert fake.register("nope")["registered"] and fake.model("nope")["id"] == "nope"  # register creates it
    assert fake.update("silero-vad", tags=["a"])["tags"] == ["a"]  # an unregistered model can be changed
    assert fake.request_lease("x:1", ["unknown", "default-vad"])["models"] == ["unknown", "silero-vad"]
    [step] = fake.plan([{"name": "a", "models": ["unknown"]}])["steps"]
    assert step["model"] == "unknown" and step["load"]
    fake.plan([{"name": "a", "models": ["silero-vad"], "keys": {"silero-vad": [KEY]}}])
    assert fake.calls[-2:] == [("plan", ("staged", 1)), ("index_lookup", ("plan", 1))]  # plan asks the index
