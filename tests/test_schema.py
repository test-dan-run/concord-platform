import pytest
from factories import base_engine_manifest, base_utility_manifest

from tools.schema import Engine, PlatformError, load_platform


def errors_of(services) -> str:
    with pytest.raises(PlatformError) as exc:
        services.load()
    return "\n".join(exc.value.errors)


def test_sample_manifests_validate():
    p = load_platform()
    assert [s.name for s in p.services] == [
        "apertus-sea-lion-v4-8b-it", "ark-asr-0-6b", "bge-m3", "cohere-transcribe-03-2026", "deepfilternet3", "dnsmos", "faster-whisper-large-v3",
        "gemma-sea-lion-v4-27b-it-nvfp4", "granite-speech-5-0-2b", "kokoro-tts", "meralion-3-3b-asr", "monkey-custom", "nemotron-3-diarization",
        "nemotronlabs-re-use",
        "pyannote-diarization", "qwen-sea-lion-v4-5-27b-it", "qwen2-5-14b-instruct-awq", "qwen3-32b-awq",
        "qwen3-4b-instruct-2507", "qwen3-5-4b", "qwen3-8b", "qwen3-asr-1-7b", "qwen3-forced-aligner", "redimnet2-sv",
        "silero-vad", "text-normalizer", "utmos", "voxcpm2", "wavlm-sv", "whisper-large-v3",
        "whisper-large-v3-turbo", "wpe-dereverb"]
    assert p.warnings == []
    aliases = {s.alias: s.name for s in p.services}
    assert aliases == {
        "default-chat": "qwen3-8b", "default-asr": "whisper-large-v3", "default-embed": "bge-m3",
        "default-tts": "kokoro-tts", "heavy-chat": "qwen3-32b-awq", "default-vad": "silero-vad",
        "default-normalizer": "text-normalizer", "fast-chat": "qwen3-4b-instruct-2507",
        "mid-chat": "qwen2-5-14b-instruct-awq", "default-vlm": "qwen3-5-4b",
        "default-diarization": "nemotron-3-diarization", None: "wpe-dereverb",
    }


def test_duplicate_port_names_both_files(services):
    services.write(base_engine_manifest(port=8201))
    msg = errors_of(services)
    assert "port 8201 collision" in msg and "qwen3-8b.yaml" in msg and "test-llm.yaml" in msg


def test_service_over_util_limit_alone_fails(services):
    services.write(base_engine_manifest(gpu={"util": 0.95}))
    assert "test-llm.yaml: needs 0.950 of the GPU, over util_limit 0.93 even when running alone" in errors_of(services)


def test_resolve_names_and_aliases_in_order_without_duplicates():
    p = load_platform()
    assert [s.name for s in p.resolve(["default-asr", "silero-vad", "whisper-large-v3"])] == ["whisper-large-v3", "silero-vad"]
    with pytest.raises(KeyError, match="unknown service 'nope'"):
        p.resolve(["nope"])


def test_alias_cannot_shadow_a_name(services):
    services.write(base_engine_manifest(alias="bge-m3"))
    assert "duplicate name/alias 'bge-m3'" in errors_of(services)


def test_custom_model_without_custom_reason_fails(services):
    services.edit("kokoro-tts", custom_reason=None)
    assert "custom_reason is required" in errors_of(services)


def test_non_openai_without_contract_reason_fails(services):
    services.edit("silero-vad", contract_reason=None)
    assert "contract_reason is required when contract is 'oip-v2'" in errors_of(services)


def test_openai_inspired_api_must_be_defined_in_contracts(services):
    services.edit("text-normalizer", api="text.summarize")
    assert "api 'text.summarize' is not valid for contract 'openai-inspired' (allowed: text.itn, text.transform)" in errors_of(services)


def test_vllm_with_non_openai_contract_fails(services):
    services.write(base_engine_manifest(contract="oip-v2", api="infer", contract_reason="x",
                                        signature={"inputs": [{"name": "X", "datatype": "FP32", "shape": [-1]}],
                                                   "outputs": [{"name": "Y", "datatype": "FP32", "shape": [-1]}]}))
    assert "contract 'oip-v2' does not match engine 'vllm' contract 'openai'" in errors_of(services)


def test_oip_without_signature_fails(services):
    services.edit("silero-vad", signature=None)
    assert "signature is required for contract 'oip-v2'" in errors_of(services)


def test_signature_forbidden_outside_oip(services):
    services.edit("text-normalizer", signature={"inputs": [{"name": "X", "datatype": "FP32", "shape": [1]}],
                                                "outputs": [{"name": "Y", "datatype": "FP32", "shape": [1]}]})
    assert "signature is only allowed for contract 'oip-v2'" in errors_of(services)


def test_invalid_oip_datatype_fails(services):
    services.edit("silero-vad", signature={"inputs": [{"name": "AUDIO", "datatype": "FLOAT32", "shape": [-1]}],
                                           "outputs": [{"name": "S", "datatype": "FP32", "shape": [-1, 2]}]})
    assert "signature.inputs.0.datatype" in errors_of(services)


def test_utility_with_custom_reason_fails(services):
    services.edit("text-normalizer", custom_reason="nope")
    assert "custom_reason is only for custom models" in errors_of(services)


def test_utility_must_be_custom(empty_services):
    empty_services.write(base_utility_manifest(engine="vllm", model="x/y", gpu={"util": 0.1}))
    assert "kind 'utility' requires engine 'custom'" in errors_of(empty_services)


def test_engine_model_must_be_a_local_copy(empty_services):
    empty_services.write(base_engine_manifest(model="Qwen/Qwen3-8B-FP8"))
    assert "must be a local copy, /models/<dir>" in errors_of(empty_services)


def test_utility_with_model_fails(empty_services):
    empty_services.write(base_utility_manifest(model="x/y"))
    assert "model is not allowed for kind 'utility'" in errors_of(empty_services)


def test_unknown_key_fails(services):
    services.edit("bge-m3", replicas=2)
    assert "replicas: Extra inputs are not permitted" in errors_of(services)


def test_tasks_are_required(services):
    services.edit("bge-m3", tasks=None)
    assert "tasks: Field required" in errors_of(services)


def test_tasks_must_be_kebab_and_unique(empty_services):
    empty_services.write(base_engine_manifest(tasks=["Tool Calling"]))
    assert "task must be kebab-case" in errors_of(empty_services)
    empty_services.write(base_engine_manifest(tasks=["chat", "chat"]))
    assert "tasks contain duplicates" in errors_of(empty_services)


def test_unknown_engine_fails(empty_services):
    empty_services.write(base_engine_manifest(engine="tgi"))
    assert "engine 'tgi' is not defined in engines.yaml" in errors_of(empty_services)


def test_engine_requires_util_and_forbids_build(empty_services):
    empty_services.write(base_engine_manifest(gpu=None, build={"context": "custom/x", "image": "a/b:1"}))
    msg = errors_of(empty_services)
    assert "engine 'vllm' requires gpu.util" in msg and "build is not allowed with engine 'vllm'" in msg


def test_custom_gpu_must_use_vram_gb(services):
    services.edit("kokoro-tts", gpu={"util": 0.1})
    assert "gpu.vram_gb, not gpu.util" in errors_of(services)


def test_gpu_needs_exactly_one_of_util_and_vram(services):
    services.edit("kokoro-tts", gpu={"util": 0.1, "vram_gb": 3})
    assert "exactly one of util" in errors_of(services)


def test_custom_requires_build(services):
    services.edit("kokoro-tts", build=None)
    assert "engine 'custom' requires build" in errors_of(services)


def test_weights_yaml_requires_weights_in_manifest(services):
    services.edit("silero-vad", weights=None)
    assert "weights.yaml exists, so the manifest needs `weights`" in errors_of(services)


def test_unknown_build_base_fails(services):
    services.edit("text-normalizer", build={"context": "custom/text-normalizer", "image": "{registry}/x:0.1.0", "base": "py-rocm"})
    assert "build.base 'py-rocm' is not in custom/_bases/bases.yaml" in errors_of(services)


def test_base_version_must_be_one_the_base_has(services):
    build = {"context": "custom/text-normalizer", "image": "{registry}/x:0.1.0", "base": "py-cpu"}
    services.edit("text-normalizer", build={**build, "base_version": "0.0.9"})
    assert "build.base_version '0.0.9' is not a version of base 'py-cpu' (known: 0.2.0, 0.1.1)" in errors_of(services)
    services.edit("text-normalizer", build={**build, "base": "none", "base_version": "0.1.1"})
    assert "build.base_version '0.1.1' is not a version of base 'none'" in errors_of(services)
    services.edit("text-normalizer", build=build)  # no pin: the base's current version
    p = services.load()
    assert p.base_image(p.get("text-normalizer")) == "ghcr.io/owner/py-cpu-base:0.2.0"


def test_file_name_must_match_name(services):
    services.write(base_engine_manifest(), filename="other.yaml")
    assert "other.yaml: file name must match name 'test-llm'" in errors_of(services)


def test_port_band_is_a_warning_not_an_error(services):
    services.write(base_engine_manifest(port=8599))
    p = services.load()
    assert any("port 8599 is outside 82xx" in w for w in p.warnings)


def test_engine_images_must_be_pinned():
    base = {"contract": "openai", "internal_port": 8000, "health_path": "/health", "command": []}
    with pytest.raises(ValueError, match="latest"):
        Engine.model_validate(base | {"image": "vllm/vllm-openai:latest"})
    with pytest.raises(ValueError, match="pin a tag"):
        Engine.model_validate(base | {"image": "vllm/vllm-openai"})


def test_kebab_case_names(empty_services):
    empty_services.write(base_engine_manifest(name="Test_LLM"), filename="Test_LLM.yaml")
    assert "name must be kebab-case" in errors_of(empty_services)


def test_compose_project_defaults_and_must_be_a_compose_name():
    from pydantic import ValidationError

    from tools.schema import Config

    base = {"gpu_total_gb": 32, "util_limit": 0.9, "assets_dir": "~/a", "registry": "ghcr.io/x"}
    assert Config.model_validate(base).compose_project == "aiplatform"
    assert Config.model_validate(base | {"compose_project": "concord-kit_2"}).compose_project == "concord-kit_2"
    with pytest.raises(ValidationError, match="compose_project"):
        Config.model_validate(base | {"compose_project": "Kit"})
