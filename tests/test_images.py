"""images.build: base variants from bases.yaml, build args, build order (docker mocked)."""

import pytest

from tools import identity, images
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


def test_up_to_date_base_is_reused_and_stale_base_rebuilt(monkeypatch, docker, services):
    services.edit("text-normalizer", build={"context": "custom/text-normalizer", "image": "{registry}/x:0.1.0", "base": "py-cpu"})
    p = services.load()  # unpinned, so on the base's current version
    s = p.get("text-normalizer")
    monkeypatch.setattr(images, "image_exists", lambda ref: True)
    current = images.base_digest(p.root / "custom" / "_bases", {"dockerfile": "Dockerfile.py-cpu"})
    monkeypatch.setattr(images, "image_label", lambda ref, label: current)
    images.build(p, s)
    assert len(docker) == 1  # only the service image
    monkeypatch.setattr(images, "image_label", lambda ref, label: "old-digest")
    docker.clear()
    images.build(p, s)
    assert len(docker) == 2 and f"aiplatform.base-digest={current}" in docker[0]


def test_pinned_older_base_is_kept_even_when_its_files_changed(monkeypatch, docker):
    p = load_platform()
    s = p.get("kokoro-tts")
    assert p.base_image(s) == "ghcr.io/owner/py-cuda-base:0.1.1"  # pinned; the base's current version is 0.2.0
    monkeypatch.setattr(images, "image_exists", lambda ref: True)
    monkeypatch.setattr(images, "image_label", lambda ref, label: "digest-of-the-0.1.1-files")
    images.build(p, s)
    assert len(docker) == 1 and "BASE_IMAGE=ghcr.io/owner/py-cuda-base:0.1.1" in docker[0]
    monkeypatch.setattr(images, "image_exists", lambda ref: False)
    docker.clear()
    images.build(p, s)  # missing: built from the current files under the old tag
    assert len(docker) == 2 and docker[0][docker[0].index("-t") + 1] == "ghcr.io/owner/py-cuda-base:0.1.1"


# Computed with the code before base versions could be pinned (bases.yaml at 0.1.1 only):
# config hash, base image, service image, and the build key for a fixed image id and weights hash.
PINNED_BEFORE = {
    "kokoro-tts": ("f75d287abd9630f5", "ghcr.io/owner/py-cuda-base:0.1.1", "ghcr.io/owner/kokoro-tts:0.1.0", "97cb8025a05ade3f"),
    "voxcpm2": ("d296b5dd18bc1c48", "ghcr.io/owner/py-cuda-torch211-base:0.1.1", "ghcr.io/owner/voxcpm2:0.1.0", "992cdcfb99cf6f3f"),
}


@pytest.mark.parametrize("name", sorted(PINNED_BEFORE))
def test_services_kept_on_the_old_base_keep_their_build_identity(name):
    p = load_platform()
    s = p.get(name)
    config = identity.config_hash(p, s)
    assert (config, p.base_image(s), p.image(s), identity.build_key("sha256:fake", config, "w")) == PINNED_BEFORE[name]


# Batching waves move services to the 0.2.0 bases (aisvc batching); kokoro-tts and voxcpm2 stay on 0.1.1.
BATCHED = {"wavlm-sv", "redimnet2-sv", "utmos", "dnsmos"}  # wave A


def test_every_custom_service_stays_on_base_0_1_1_until_its_wave():
    p = load_platform()
    versions = {s.name: p.base_version(s) for s in p.services if s.is_custom and s.base != "none"}
    assert {n for n, v in versions.items() if v == "0.2.0"} == BATCHED
    assert {v for n, v in versions.items() if n not in BATCHED} == {"0.1.1"}
    assert versions["kokoro-tts"] == versions["voxcpm2"] == "0.1.1"
