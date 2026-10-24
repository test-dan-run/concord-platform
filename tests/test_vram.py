import pytest
from factories import base_engine_manifest, base_utility_manifest

from tools import vram_check
from tools.schema import load_platform

VOICE_CHAT = ["qwen3-8b", "whisper-large-v3", "kokoro-tts", "bge-m3", "silero-vad", "text-normalizer"]


def test_sample_sets():
    p = load_platform()
    voice = vram_check.budget(p, p.resolve(VOICE_CHAT))
    assert voice.total_fraction == pytest.approx(0.674, abs=5e-4)  # the sum of minimums, still reported
    assert not voice.ok and voice.gpu == ["qwen3-8b", "whisper-large-v3", "kokoro-tts", "bge-m3"]  # largest first
    assert vram_check.budget(p, p.resolve(["heavy-chat"])).ok
    assert vram_check.budget(p, p.resolve(["silero-vad", "text-normalizer"])).total_fraction == 0
    # one GPU service per stage: an ASR model with CPU diarisation and VAD is fine; two ASR models are not
    assert vram_check.budget(p, p.resolve(["whisper-large-v3", "nemotron-3-diarization", "silero-vad"])).ok
    assert not vram_check.budget(p, p.resolve(["whisper-large-v3", "faster-whisper-large-v3"])).ok


def test_two_gpu_services_are_refused(empty_services, capsys):
    empty_services.write(base_engine_manifest(name="big", port=8291, gpu={"util": 0.3}))
    empty_services.write(base_engine_manifest(name="small", port=8292, gpu={"util": 0.2}))
    empty_services.write(base_utility_manifest())
    p = empty_services.load()
    b = vram_check.budget(p, p.resolve(["big", "small"]))
    assert b.total_fraction == pytest.approx(0.5) and not b.ok  # would have fit side by side
    assert vram_check.main(["big", "small"], platform=p) == 1
    out = capsys.readouterr()
    assert "REFUSED" in out.out and "error: big, small are 2 GPU services; only one GPU service runs at a time" in out.err
    assert vram_check.main(["big", "text-normalizer"], platform=p) == 0
    assert "needs at least 0.300 of 32 GB" in capsys.readouterr().out


def test_cpu_only_services_count_as_zero(empty_services, capsys):
    empty_services.write(base_engine_manifest(gpu={"util": 0.5}))
    empty_services.write(base_utility_manifest())
    p = empty_services.load()
    b = vram_check.budget(p, p.services)
    assert b.total_fraction == pytest.approx(0.5) and b.ok
    cpu = [r for r in b.rows if r.source == "cpu"]
    assert [r.name for r in cpu] == ["text-normalizer"] and cpu[0].fraction == 0 and cpu[0].gb == 0
    assert vram_check.main([], platform=p) == 0
    assert "CPU" in capsys.readouterr().out


def test_unknown_name(capsys):
    assert vram_check.main(["nope"], platform=load_platform()) == 2
    assert "unknown service 'nope'" in capsys.readouterr().err


def test_custom_vram_gb_is_converted_to_fraction():
    p = load_platform()
    rows = {r.name: r for r in vram_check.budget(p, p.resolve(VOICE_CHAT)).rows}
    assert rows["kokoro-tts"].fraction == pytest.approx(3 / 32) and rows["kokoro-tts"].source == "vram_gb"


# Memory at start: min(util_limit, (free - 1 GiB) / total), measured right before the start.

def test_start_memory_takes_what_is_free_less_one_gib_up_to_util_limit():
    p = load_platform()
    assert vram_check.start_memory(p, (31.84, 26.0)) == vram_check.StartMemory(0.785, 24.99, True)  # 25 / 31.84
    assert vram_check.start_memory(p, (31.84, 31.84)).fraction == 0.93  # capped at util_limit
    assert vram_check.start_memory(p, (31.84, 0.5)).fraction == 0.0
    assert vram_check.start_memory(p, (31.84, 26.0)).env() == {"AIPLATFORM_GPU_UTIL": "0.785", "AIPLATFORM_VRAM_GB": "24.99"}


def test_without_a_measurement_start_memory_is_util_limit_of_the_configured_gpu():
    p = load_platform()
    assert vram_check.start_memory(p, None) == vram_check.StartMemory(0.93, 0.93 * 32, False)
    assert vram_check.start_env(p, p.get("heavy-chat"), None) == {"AIPLATFORM_GPU_UTIL": "0.930", "AIPLATFORM_VRAM_GB": "29.76"}


def test_start_env_refuses_when_the_minimum_is_not_free():
    p = load_platform()
    assert vram_check.start_env(p, p.get("silero-vad"), (31.84, 0.0)) == {}  # CPU: nothing to size
    with pytest.raises(ValueError, match="whisper-large-v3 needs at least 4.8 GiB of the GPU to load; 5.0 GiB is free"):
        vram_check.start_env(p, p.get("whisper-large-v3"), (31.84, 5.0))  # 4.8 + 1 GiB left free
    assert vram_check.start_env(p, p.get("whisper-large-v3"), (31.84, 6.0))["AIPLATFORM_GPU_UTIL"] == "0.157"
