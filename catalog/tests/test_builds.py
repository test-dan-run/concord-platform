"""Build identity: what changes a build key and what does not."""

import dataclasses
import time

from catalog_helpers import wait_builds

from tools import identity, render
from tools.schema import load_platform


def test_config_hash_ignores_fields_that_do_not_change_outputs():
    p = load_platform()
    s = p.get("whisper-large-v3")
    base = identity.config_hash(p, s)
    same = s.model_copy(update={"port": 8199, "description": "x", "alias": None, "tasks": ["other"]})
    assert identity.config_hash(p, same) == base
    assert identity.config_hash(p, s.model_copy(update={"args": ["--max-model-len=224"]})) != base
    # GPU size is not part of the build: the model's share, nor the device size and util_limit
    bigger = s.model_copy(update={"gpu": s.gpu.model_copy(update={"util": 0.5})})
    smaller_gpu = dataclasses.replace(p, config=p.config.model_copy(update={"gpu_total_gb": 16, "util_limit": 0.8}))
    assert identity.config_hash(p, bigger) == identity.config_hash(smaller_gpu, s) == base


def test_containers_carry_their_config_hash():
    p = load_platform()
    labels = render.compose_service(p, p.get("silero-vad"), p.root)["labels"]
    assert labels["aiplatform.config_hash"] == identity.config_hash(p, p.get("silero-vad"))


def test_weights_yaml_identifies_custom_weights():
    p = load_platform()
    src = identity.weights_source(p, p.get("silero-vad"))
    assert src.kind == "yaml" and len(identity.weights_yaml_hash(src.path)) == 16


def test_build_key_follows_image_and_weights(cat, runtime, assets):
    wait_builds(cat, ["whisper-large-v3"])
    first = cat.scheduler.runs["whisper-large-v3"].build
    runtime.images["vllm/vllm-openai:v0.31.0"] = "sha256:rebuilt"
    wait_builds(cat, ["whisper-large-v3"])
    assert cat.scheduler.runs["whisper-large-v3"].build.build_key != first.build_key  # same tag, new image
    runtime.images.clear()
    wait_builds(cat, ["whisper-large-v3"])
    assert cat.scheduler.runs["whisper-large-v3"].build.build_key == first.build_key
    (assets / "whisper-large-v3" / "SOURCE.txt").write_text("repo: y\n")  # provenance notes are not weights
    cat.builds.hasher._quick.clear()
    wait_builds(cat, ["whisper-large-v3"])
    assert cat.scheduler.runs["whisper-large-v3"].build.build_key == first.build_key
    time.sleep(0.01)
    (assets / "whisper-large-v3" / "model.safetensors").write_bytes(b"new weights")
    cat.builds.hasher._quick.clear()
    deadline = time.monotonic() + 10
    while cat.scheduler.runs["whisper-large-v3"].build.build_key in (None, first.build_key):
        assert time.monotonic() < deadline
        cat.scheduler.tick()
        time.sleep(0.02)
    assert cat.db.one("SELECT COUNT(*) AS n FROM build WHERE model_id = 'whisper-large-v3'")["n"] == 3
