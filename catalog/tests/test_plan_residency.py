"""A job plan counts a running GPU model as loaded only while no earlier step has loaded another GPU model (one GPU
model runs at a time), and a stage that names one model twice (its alias and its id) gives it one step."""

import pytest
from catalog_helpers import step
from fastapi.testclient import TestClient

from catalog.app import create_app


@pytest.fixture
def api(cat):
    with TestClient(create_app(cat, start=False), raise_server_exceptions=False) as c:
        yield c


def run(cat, name: str) -> None:
    cat.scheduler.ensure_started(name, holder="ui")
    step(cat)
    assert cat.scheduler.model_view(name)["state"] == "running"


def loads(plan: dict) -> list[tuple[str, bool]]:
    return [(s["model"], s["load"]) for s in plan["steps"]]


def test_a_running_gpu_model_loads_again_after_an_earlier_step_loaded_another(api, cat):
    run(cat, "whisper-large-v3")
    body = {"stages": [{"name": "seg", "models": ["pyannote-diarization"]},
                       {"name": "asr", "models": ["whisper-large-v3"], "after": "seg"}]}
    plan = api.post("/api/jobs", json=body).json()
    assert loads(plan) == [("pyannote-diarization", True), ("whisper-large-v3", True)]
    assert plan["loads"] == 2 and plan["est_load_s"] == round(sum(s["cold_start_s"] for s in plan["steps"]), 1)


def test_a_running_gpu_model_is_loaded_while_it_is_still_the_one_up(api, cat):
    run(cat, "whisper-large-v3")
    body = {"stages": [{"name": "asr", "models": ["whisper-large-v3"]},
                       {"name": "seg", "models": ["pyannote-diarization", "silero-vad"], "after": "asr"}]}
    plan = api.post("/api/jobs", json=body).json()
    assert loads(plan) == [("whisper-large-v3", False), ("silero-vad", True), ("pyannote-diarization", True)]
    assert plan["loads"] == 2


def test_a_running_cpu_model_stays_loaded_after_a_gpu_model_loads(api, cat):
    run(cat, "silero-vad")
    body = {"stages": [{"name": "seg", "models": ["pyannote-diarization"]},
                       {"name": "vad", "models": ["silero-vad"], "after": "seg"}]}
    plan = api.post("/api/jobs", json=body).json()
    assert loads(plan) == [("pyannote-diarization", True), ("silero-vad", False)] and plan["loads"] == 1


def test_a_co_resident_plan_reloads_a_running_gpu_model_another_gpu_model_evicted(api, cat):
    run(cat, "pyannote-diarization")
    body = {"pipeline": "co-resident",
            "stages": [{"name": "seg", "models": ["pyannote-diarization"]},
                       {"name": "asr", "models": ["qwen3-asr-1-7b", "whisper-large-v3"], "after": "seg"}]}
    plan = api.post("/api/jobs", json=body).json()
    seg = [s for s in plan["steps"] if s["model"] == "pyannote-diarization"]
    assert seg and all(s["load"] and s["alongside"] for s in seg), plan["steps"]  # the inner loop, after an ASR load


@pytest.mark.parametrize("pipeline", ["staged", "co-resident"])
def test_a_stage_naming_a_model_by_alias_and_id_gives_it_one_step(api, pipeline):
    keys = [{"audio_sha256": f"{i:064x}", "span_start": -1, "span_end": -1, "params_hash": "p" * 16} for i in range(2)]
    body = {"pipeline": pipeline,
            "stages": [{"name": "seg", "models": ["silero-vad", "default-vad"], "keys": {"default-vad": keys}},
                       {"name": "asr", "models": ["default-asr", "whisper-large-v3"], "after": "seg"}]}
    r = api.post("/api/jobs", json=body)
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert {s["model"] for s in steps} == {"silero-vad", "whisper-large-v3"}  # aliases in, names out
    per_stage = [(s["stage"], s["model"], s.get("alongside")) for s in steps]
    assert len(per_stage) == len(set(per_stage)), per_stage
    assert {s["items"] for s in steps if s["model"] == "silero-vad"} == {2}  # the keys given under the alias count
    assert r.json()["loads"] == 2
