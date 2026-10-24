import pytest
from factories import base_engine_manifest, base_utility_manifest

from tools import vram_check
from tools.schema import load_platform

VOICE_CHAT = ["qwen3-8b", "whisper-large-v3", "kokoro-tts", "bge-m3", "silero-vad", "text-normalizer"]


def test_sample_sets():
    p = load_platform()
    assert vram_check.budget(p, p.resolve(VOICE_CHAT)).total_fraction == pytest.approx(0.674, abs=5e-4)
    assert vram_check.budget(p, p.resolve(["heavy-chat"])).total_fraction == pytest.approx(0.85)
    assert vram_check.budget(p, p.resolve(["silero-vad", "text-normalizer"])).total_fraction == 0
    # the Concord benchmark set: both Whisper services and both GPU/CPU diarisers fit together
    asr_diar = p.resolve(["whisper-large-v3", "faster-whisper-large-v3", "pyannote-diarization",
                          "nemotron-3-diarization", "silero-vad"])
    assert vram_check.budget(p, asr_diar).ok


def test_set_over_limit_fails(empty_services, capsys):
    empty_services.write(base_engine_manifest(name="big", port=8291, gpu={"util": 0.6}))
    empty_services.write(base_engine_manifest(name="small", port=8292, gpu={"util": 0.35}))
    p = empty_services.load()
    b = vram_check.budget(p, p.resolve(["big", "small"]))
    assert b.total_fraction == pytest.approx(0.95) and not b.ok
    assert vram_check.main(["big", "small"], platform=p) == 1
    assert "OVER BUDGET" in capsys.readouterr().out
    assert vram_check.main(["big"], platform=p) == 0


def test_cpu_only_services_count_as_zero(empty_services, capsys):
    empty_services.write(base_engine_manifest(gpu={"util": 0.5}))
    empty_services.write(base_utility_manifest())
    p = empty_services.load()
    b = vram_check.budget(p, p.services)
    assert b.total_fraction == pytest.approx(0.5)
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
