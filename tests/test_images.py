"""images.build: base variants from bases.yaml, build args, build order (docker mocked)."""

import pytest

from tools import images
from tools.schema import load_platform


@pytest.fixture
def docker(monkeypatch):
    runs = []
    monkeypatch.setattr(images, "_run", lambda cmd: runs.append(cmd))
    monkeypatch.setattr(images, "image_exists", lambda ref: False)
    monkeypatch.setattr(images, "image_label", lambda ref, label: None)
    return runs


def test_base_variant_passes_its_build_args(docker):
    p = load_platform()
    images.build(p, p.get("pyannote-diarization"))
    base, service = docker
    assert base[base.index("-f") + 1].endswith("custom/_bases/Dockerfile.py-cuda")
    args = [base[i + 1] for i, a in enumerate(base) if a == "--build-arg"]
    assert args == ["TORCH_EXTRA=torchaudio==2.11.0", "TORCH_VERSION=2.11.0"]
    assert base[base.index("-t") + 1] == "ghcr.io/owner/py-cuda-torch211-base:0.1.1"
    assert service[service.index("--build-arg") + 1] == "BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.1.1"
    assert service[-1].endswith("custom/pyannote-diarization")


def test_plain_base_has_no_build_args(docker):
    p = load_platform()
    images.build(p, p.get("faster-whisper-large-v3"))  # GPU service on the slim base (pip CUDA wheels)
    assert "--build-arg" not in docker[0] and docker[0][-1].endswith("custom/_bases")
    assert "BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1" in docker[1]


def test_engine_services_have_nothing_to_build(docker):
    p = load_platform()
    with pytest.raises(SystemExit, match="upstream vllm image"):
        images.build(p, p.get("qwen3-5-4b"))


def test_up_to_date_base_is_reused_and_stale_base_rebuilt(monkeypatch, docker):
    p = load_platform()
    s = p.get("text-normalizer")
    current = images.base_digest(p.root / "custom" / "_bases", {"dockerfile": "Dockerfile.py-cpu"})
    monkeypatch.setattr(images, "image_label", lambda ref, label: current)
    images.build(p, s)
    assert len(docker) == 1  # only the service image
    monkeypatch.setattr(images, "image_label", lambda ref, label: "old-digest")
    docker.clear()
    images.build(p, s)
    assert len(docker) == 2 and f"aiplatform.base-digest={current}" in docker[0]
