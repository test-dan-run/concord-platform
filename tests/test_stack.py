"""stack.up ordering, one GPU service at a time, memory at start and leases, with Docker mocked out."""

import json
import threading
import time

import pytest

from tools import stack
from tools.leases import Lease, Leases


@pytest.fixture
def docker(monkeypatch, tmp_path):
    calls = {"compose": [], "waited": [], "env": {}, "gpu": None}

    def compose(p, *args, env=None, **kw):
        calls["compose"].append(args)
        if args[0] == "up":
            calls["env"][args[-1]] = env

    monkeypatch.setattr(stack, "LEASES_DIR", tmp_path / "run" / "leases")
    monkeypatch.setattr(stack, "require_docker", lambda: None)
    monkeypatch.setattr(stack, "ensure_network", lambda name: None)
    monkeypatch.setattr(stack, "ensure_assets", lambda p, s: None)
    monkeypatch.setattr(stack.images, "image_exists", lambda ref: True)
    monkeypatch.setattr(stack, "measure_gpu", lambda: calls["gpu"])  # (total, free) GiB; None: no nvidia-smi
    monkeypatch.setattr(stack, "compose", compose)
    monkeypatch.setattr(stack, "wait_healthy", lambda p, name, timeout: calls["waited"].append(name))
    return calls


def running(**services):
    return {n: stack.Container("id", n, "running", health, gpu) for n, (gpu, health) in services.items()}


def started(docker):
    return [a[-1] for a in docker["compose"] if a[0] == "up"]


def test_up_starts_gpu_first_and_skips_healthy(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"text-normalizer": (False, "healthy")}))
    stack.up(["default-vad", "default-chat", "default-normalizer"], timeout=5)
    assert started(docker) == ["qwen3-8b", "silero-vad"]
    assert docker["waited"] == started(docker)  # each one health-gated before the next
    assert all(a[:4] == ("up", "-d", "--no-deps", "--no-build") for a in docker["compose"])
    # No nvidia-smi: the GPU service gets util_limit x gpu_total_gb; CPU services get nothing.
    assert docker["env"] == {"qwen3-8b": {"AIPLATFORM_GPU_UTIL": "0.930", "AIPLATFORM_VRAM_GB": "29.76"}, "silero-vad": {}}


def test_the_gpu_service_gets_the_memory_measured_free(docker, monkeypatch, capsys):
    monkeypatch.setattr(stack, "containers", lambda p: {})
    docker["gpu"] = (31.84, 30.2)  # Windows keeps some for the display under WSL2
    stack.up(["default-tts"], timeout=5)
    assert docker["env"]["kokoro-tts"] == {"AIPLATFORM_GPU_UTIL": "0.917", "AIPLATFORM_VRAM_GB": "29.19"}  # (30.2 - 1) / 31.84
    assert "kokoro-tts: 0.917 of the GPU (30.2 of 31.8 GiB free)" in capsys.readouterr().out


def test_a_gpu_service_whose_minimum_is_not_free_is_refused(docker, monkeypatch):
    docker["gpu"] = (31.84, 20.0)  # heavy-chat needs 0.85 x 31.84 = 27.1 GiB
    p = stack.prepare()
    with pytest.raises(stack.StackError, match="qwen3-32b-awq needs at least 27.1 GiB of the GPU to load; 20.0 GiB is free"):
        stack.gpu_env(p, p.get("heavy-chat"), wait_s=0)
    docker["gpu"] = (31.84, 29.0)
    assert stack.gpu_env(p, p.get("heavy-chat"), wait_s=0)["AIPLATFORM_GPU_UTIL"] == "0.879"


def test_a_cpu_request_leaves_the_running_gpu_service_alone(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    stack.up(["silero-vad", "text-normalizer"], timeout=5)
    assert not any(a[0] == "stop" for a in docker["compose"])
    assert started(docker) == ["silero-vad", "text-normalizer"]


def test_other_running_gpu_services_are_stopped_cpu_kept(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: running(**{
        "qwen3-8b": (True, "healthy"), "bge-m3": (True, "healthy"),
        "silero-vad": (False, "healthy"), "text-normalizer": (False, "healthy"),
    }))
    stack.up(["heavy-chat"], timeout=5)
    assert docker["compose"][0] == ("stop", "bge-m3", "qwen3-8b")
    assert started(docker) == ["qwen3-32b-awq"]


def test_no_evict_refuses_when_another_gpu_service_runs(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"bge-m3": (True, "healthy")}))
    with pytest.raises(stack.StackError, match="bge-m3 is running on the GPU and only one GPU service runs at a time"):
        stack.up(["default-asr"], timeout=5, evict=False)  # 0.08 + 0.15 would have fit side by side
    assert docker["compose"] == []


def test_running_gpu_service_without_a_manifest_is_stopped(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"old-model": (True, "healthy")}))
    stack.up(["default-asr"], timeout=5)
    assert docker["compose"][0] == ("stop", "old-model")


def test_two_gpu_services_are_refused(docker, monkeypatch, capsys):
    monkeypatch.setattr(stack, "containers", lambda p: {})
    with pytest.raises(stack.StackError, match="only one GPU service runs at a time"):
        stack.up(["default-asr", "default-embed", "silero-vad"], timeout=5)  # small enough to have packed
    assert stack.main(["up", "default-asr", "pyannote-diarization"]) == 1
    assert "error: pyannote-diarization, whisper-large-v3 are 2 GPU services" in capsys.readouterr().err
    assert docker["compose"] == []


def test_unknown_service(docker):
    with pytest.raises(stack.StackError, match="unknown service 'nope'"):
        stack.up(["nope"], timeout=5)


def test_down_named_services(docker):
    stack.down(["default-vad", "pyannote-diarization"])
    assert docker["compose"] == [("rm", "--stop", "--force", "silero-vad", "pyannote-diarization")]
    stack.down([])
    assert docker["compose"][-1] == ("down",)


# --- leases ---------------------------------------------------------------------------------------

def store() -> Leases:
    return Leases(stack.LEASES_DIR)


def stopped(docker):
    return [n for a in docker["compose"] if a[0] == "stop" for n in a[1:]]


def test_up_with_holder_writes_a_lease(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: {})
    stack.up(["default-asr", "default-vad"], timeout=5, holder="bench:job-12", ttl=60)
    [f] = stack.LEASES_DIR.glob("*.json")
    d = json.loads(f.read_text())
    assert d["holder"] == "bench:job-12"
    assert d["services"] == ["silero-vad", "whisper-large-v3"]  # aliases resolved
    assert 50 < d["expires_at"] - time.time() <= 60


def test_leased_service_is_not_evicted_and_the_request_is_refused(docker, monkeypatch, capsys):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    assert stack.main(["up", "heavy-chat", "--holder", "label:job-7"]) == 1
    assert docker["compose"] == []
    err = capsys.readouterr().err
    assert "error: the GPU is leased to qwen3-8b (held by bench:job-1)" in err
    assert set(store().live()) == {"bench:job-1"}  # the refused request left no lease


def test_up_without_holder_also_respects_leases(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    with pytest.raises(stack.StackError, match=r"leased to qwen3-8b \(held by bench:job-1\)"):
        stack.up(["heavy-chat"], timeout=5)
    assert docker["compose"] == []


def test_cpu_services_start_next_to_a_leased_gpu_service(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    stack.up(["silero-vad"], timeout=5, holder="label:job-7")
    assert stopped(docker) == [] and started(docker) == ["silero-vad"]


def test_an_unleased_gpu_service_is_stopped_next_to_leased_cpu_services(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["silero-vad"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{
        "qwen3-8b": (True, "healthy"), "silero-vad": (False, "healthy")}))
    stack.up(["qwen2-5-14b-instruct-awq"], timeout=5)
    assert stopped(docker) == ["qwen3-8b"] and started(docker) == ["qwen2-5-14b-instruct-awq"]


def test_leased_service_that_is_still_starting_counts(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: {})  # its holder has not started it yet
    with pytest.raises(stack.StackError, match="the GPU is leased to qwen3-8b"):
        stack.up(["heavy-chat"], timeout=5)


def test_shared_service_is_fine(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    stack.up(["qwen3-8b"], timeout=5, holder="label:job-7")
    assert docker["compose"] == []
    assert store().holders_by_service() == {"qwen3-8b": ["bench:job-1", "label:job-7"]}


def test_expired_lease_does_not_protect(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() - 1))
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    stack.up(["heavy-chat"], timeout=5, holder="label:job-7")
    assert stopped(docker) == ["qwen3-8b"]
    assert set(store().live()) == {"label:job-7"}
    assert len(list(stack.LEASES_DIR.glob("*.json"))) == 1  # the expired file was cleaned up


def test_lease_renews(docker, capsys):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 5))
    assert stack.main(["lease", "--holder", "bench:job-1", "--ttl", "300"]) == 0
    assert 290 < store().live()["bench:job-1"].expires_at - time.time() <= 300
    assert stack.main(["lease", "--holder", "nobody"]) == 1
    assert "error: no live lease for nobody" in capsys.readouterr().err


def test_release_frees(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    store().write(Lease("bench:job-2", ["silero-vad"], time.time() + 60))  # CPU: takes no VRAM
    assert stack.main(["release", "--holder", "bench:job-1"]) == 0
    assert set(store().live()) == {"bench:job-2"}
    monkeypatch.setattr(stack, "containers", lambda p: running(**{"qwen3-8b": (True, "healthy")}))
    stack.up(["heavy-chat"], timeout=5)
    assert stopped(docker) == ["qwen3-8b"]


def test_failed_start_removes_only_the_new_lease(docker, monkeypatch):
    store().write(Lease("bench:job-1", ["silero-vad"], time.time() + 60))
    monkeypatch.setattr(stack, "containers", lambda p: {})

    def fail(p, name, timeout):
        raise stack.StackError(f"{name} failed to start (exited/-)")

    monkeypatch.setattr(stack, "wait_healthy", fail)
    with pytest.raises(stack.StackError, match="failed to start"):
        stack.up(["default-asr", "silero-vad"], timeout=5, holder="bench:job-1")
    assert store().live()["bench:job-1"].services == ["silero-vad"]
    with pytest.raises(stack.StackError):
        stack.up(["default-asr"], timeout=5, holder="label:job-7")
    assert store().live().keys() == {"bench:job-1"}


def test_up_keeps_its_lease_alive_while_a_slow_service_starts(docker, monkeypatch):
    monkeypatch.setattr(stack, "containers", lambda p: {})
    seen = []

    def slow(p, name, timeout):
        seen.append(store().live()["bench:job-1"].expires_at)
        time.sleep(1.3)  # renewals run every ttl/3 = 1 s
        seen.append(store().live()["bench:job-1"].expires_at)

    monkeypatch.setattr(stack, "wait_healthy", slow)
    stack.up(["bge-m3"], timeout=5, holder="bench:job-1", ttl=3)
    assert seen[1] > seen[0]


def test_down_warns_about_leased_services(docker, capsys):
    store().write(Lease("bench:job-1", ["qwen3-8b"], time.time() + 60))
    stack.down(["default-chat"])
    assert "warning: stopping leased services: qwen3-8b (held by bench:job-1)" in capsys.readouterr().out
    assert docker["compose"] == [("rm", "--stop", "--force", "qwen3-8b")]


def test_leases_command_lists_live_leases(docker, capsys):
    assert stack.main(["leases"]) == 0
    assert capsys.readouterr().out.strip() == "no leases"
    store().write(Lease("bench:job-1", ["bge-m3", "qwen3-8b"], time.time() + 90))
    store().write(Lease("label:job-2", ["old"], time.time() - 1))
    stack.main(["leases"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["SERVICE", "HOLDER", "EXPIRES", "IN"]
    assert [line.split()[:2] for line in out[1:]] == [["bge-m3", "bench:job-1"], ["qwen3-8b", "bench:job-1"]]


def test_lease_store_lock_serialises_writers(tmp_path):
    s = Leases(tmp_path)
    done = threading.Event()

    def writer():
        with s.locked():
            s.add("label:job-1", ["bge-m3"], 60)
        done.set()

    with s.locked():
        t = threading.Thread(target=writer)
        t.start()
        assert not done.wait(0.3)  # blocked while another caller holds the lock
    t.join(5)
    assert done.is_set() and set(s.live()) == {"label:job-1"}


def test_holder_names_map_to_distinct_files(tmp_path):
    s = Leases(tmp_path)
    s.write(Lease("bench:job-1", ["a"], time.time() + 60))
    s.write(Lease("bench_job-1", ["b"], time.time() + 60))
    assert {h: lease.services for h, lease in s.live().items()} == {"bench:job-1": ["a"], "bench_job-1": ["b"]}


def test_commands_go_through_a_running_catalog(monkeypatch, capsys):
    from aicore.catalog import FakeCatalog

    fake = FakeCatalog([{"id": "silero-vad", "state": "stopped", "leased_by": [], "state_detail": None},
                        {"id": "whisper-large-v3", "state": "running", "leased_by": ["bench:job-1"], "state_detail": None}])
    monkeypatch.setattr(stack, "catalog", lambda: fake)
    monkeypatch.setattr(stack, "prepare", lambda: __import__("tools.schema", fromlist=["x"]).load_platform())
    assert stack.main(["up", "default-asr", "default-tts"]) == 1  # refused before the catalog is asked
    assert "only one GPU service runs at a time" in capsys.readouterr().err
    assert stack.main(["up", "default-vad"]) == 0
    assert fake.model("silero-vad")["state"] == "running"
    assert stack.main(["up", "silero-vad", "--holder", "synth:run-1"]) == 0
    assert stack.main(["lease", "--holder", "synth:run-1"]) == 0
    assert stack.main(["release", "--holder", "synth:run-1"]) == 0 and fake.leases() == []
    assert stack.main(["down", "whisper-large-v3"]) == 0
    out = capsys.readouterr().out
    assert "warning: stopping leased service whisper-large-v3 (held by bench:job-1)" in out
    assert fake.model("whisper-large-v3")["state"] == "stopped"
