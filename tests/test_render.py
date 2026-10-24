import json
import re
import shutil
import subprocess

import pytest
import yaml
from factories import base_engine_manifest

from tools import render
from tools.schema import load_platform


def compose_of(p):
    return render.build_compose(p, p.root / "compose" / "generated")


def test_vllm_and_sglang_use_their_own_memory_flag(empty_services):
    empty_services.write(base_engine_manifest(name="on-vllm", port=8291, gpu={"util": 0.3}))
    empty_services.write(base_engine_manifest(name="on-sglang", port=8292, engine="sglang", gpu={"util": 0.25}))
    svcs = compose_of(empty_services.load())["services"]
    v, s = svcs["on-vllm"]["command"], svcs["on-sglang"]["command"]
    assert v[v.index("--gpu-memory-utilization") + 1] == "0.3" and "--mem-fraction-static" not in v
    assert s[s.index("--mem-fraction-static") + 1] == "0.25" and "--gpu-memory-utilization" not in s
    assert s[:3] == ["python3", "-m", "sglang.launch_server"] and s[s.index("--model-path") + 1] == "/models/test-llm"
    assert v[v.index("--served-model-name") + 1] == "on-vllm"


def test_manifest_args_are_appended_after_template():
    cmd = compose_of(load_platform())["services"]["bge-m3"]["command"]
    assert cmd[-2:] == ["--runner=pooling", "--max-model-len=8192"]


def test_gpu_reservation_only_when_manifest_has_gpu():
    svcs = compose_of(load_platform())["services"]
    for name in ("silero-vad", "text-normalizer"):
        assert "deploy" not in svcs[name]
    for name in ("qwen3-8b", "kokoro-tts", "bge-m3"):
        dev = svcs[name]["deploy"]["resources"]["reservations"]["devices"]
        assert dev == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]


def test_ports_bind_to_localhost_only():
    for s in compose_of(load_platform())["services"].values():
        assert all(p.startswith("127.0.0.1:") for p in s["ports"])


def test_engine_service_shape():
    p = load_platform()
    s = compose_of(p)["services"]["whisper-large-v3"]
    assert s["image"] == "vllm/vllm-openai:v0.31.0" and s["shm_size"] == "16g"
    assert s["environment"] == {"HF_HUB_OFFLINE": "1"}  # local copies only: nothing is downloaded at start
    assert s["volumes"] == ["${AI_ASSETS:?set AI_ASSETS to the weights directory}:/models:ro"]
    assert s["labels"]["aiplatform.image"] == "vllm/vllm-openai:v0.31.0"
    assert set(s["labels"]) >= {"aiplatform.name", "aiplatform.kind", "aiplatform.engine", "aiplatform.contract", "aiplatform.image"}
    assert "build" not in s and "--max-model-len=448" in s["command"]


def test_no_engine_service_mounts_a_hub_cache():
    for s in compose_of(load_platform())["services"].values():
        assert not any("hf-cache" in v for v in s.get("volumes", [])) and "HF_HOME" not in s.get("environment", {})


def test_custom_service_shape():
    p = load_platform()
    s = compose_of(p)["services"]["kokoro-tts"]
    assert s["build"]["context"] == "../../custom/kokoro-tts"
    assert (p.root / "compose" / "generated" / s["build"]["context"] / "Dockerfile").is_file()
    assert s["build"]["args"]["BASE_IMAGE"] == "ghcr.io/owner/py-cuda-base:0.1.1"
    assert s["image"] == "ghcr.io/owner/kokoro-tts:0.1.0"
    assert s["volumes"] == ["${AI_ASSETS:?set AI_ASSETS to the weights directory}/custom-models/kokoro-tts/v1.0:/models:ro"]
    assert s["environment"]["AISVC_VRAM_GB"] == "3" and s["ports"] == ["127.0.0.1:8103:8000"]
    assert compose_of(p)["services"]["silero-vad"]["build"]["args"]["BASE_IMAGE"] == "ghcr.io/owner/py-cpu-base:0.1.1"


def test_healthcheck_uses_python_not_curl():
    hc = compose_of(load_platform())["services"]["qwen3-8b"]["healthcheck"]
    assert hc["test"][:3] == ["CMD", "python3", "-c"] and "http://127.0.0.1:8000/health" in hc["test"][3]


def test_dollar_in_args_is_escaped(empty_services):
    empty_services.write(base_engine_manifest(args=["--chat-template=$HOME/t.jinja"]))
    assert "--chat-template=$$HOME/t.jinja" in compose_of(empty_services.load())["services"]["test-llm"]["command"]


def test_render_is_deterministic():
    first = render.outputs(load_platform())
    second = render.outputs(load_platform())
    assert first.keys() == second.keys()
    for path in first:
        assert first[path].encode() == second[path].encode(), path


def test_endpoints_map_has_names_aliases_and_signature():
    p = load_platform()
    eps = render.build_endpoints(p)["services"]
    assert len(eps) == len(p.services) + sum(1 for s in p.services if s.alias)
    assert eps["default-vad"] == eps["silero-vad"]
    vad = eps["silero-vad"]
    assert (vad["base_path"], vad["host_port"], vad["internal_port"], vad["contract"]) == ("/v2", 8401, 8000, "oip-v2")
    assert vad["signature"]["inputs"][0] == {"name": "AUDIO", "datatype": "FP32", "shape": [-1]}
    chat = eps["default-chat"]
    assert chat["name"] == "qwen3-8b" and chat["image"] == "vllm/vllm-openai:v0.31.0" and chat["base_path"] == "/v1"
    assert "signature" not in chat and chat["labels"]["aiplatform.engine"] == "vllm"
    assert eps["text-normalizer"]["base_path"] == "/v1"


def test_endpoints_carry_description_tasks_and_gpu_share():
    eps = render.build_endpoints(load_platform())["services"]
    asr = eps["faster-whisper-large-v3"]
    assert asr["description"].startswith("Speech-to-text with word timestamps")
    assert asr["tasks"] == ["speech-recognition", "speech-translation", "language-identification", "voice-activity-detection"]
    assert asr["gpu_fraction"] == pytest.approx(5 / 32, abs=1e-4) and eps["silero-vad"]["gpu_fraction"] == 0
    assert all("profiles" not in e and "aiplatform.profiles" not in e["labels"] for e in eps.values())


def test_endpoints_yaml_has_no_anchors():
    text = render.dump_yaml(render.build_endpoints(load_platform()))
    assert "&id" not in text and "*id" not in text


def test_catalog_embeds_all_services():
    html = render.render_catalog(load_platform())
    raw = re.search(r'<script id="catalog-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw.replace("<\\/", "</"))
    assert sorted(s["name"] for s in data["services"]) == sorted(s.name for s in load_platform().services)
    vad = next(s for s in data["services"] if s["name"] == "silero-vad")
    assert vad["signature"]["outputs"][0]["name"] == "SEGMENTS" and vad["gpu"] is None
    assert vad["tasks"] == ["voice-activity-detection"]
    assert [g["name"] for g in data["gpu"]][:2] == ["qwen3-32b-awq", "gemma-sea-lion-v4-27b-it-nvfp4"]  # largest first
    assert "silero-vad" not in {g["name"] for g in data["gpu"]}
    assert "profiles" not in data and all("profiles" not in s for s in data["services"])


def test_catalog_makes_no_external_requests():
    html = render.render_catalog(load_platform())
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=", html)


def test_snippet_is_standalone():
    p = load_platform()
    for name in ("qwen3-8b", "kokoro-tts"):
        text = render.snippet(p, name)
        doc = yaml.safe_load(text)
        svc = doc["services"][name]
        assert "build" not in svc and "profiles" not in svc
        assert doc["networks"]["ai-net"] == {"name": "ai-net"}  # created by compose, not external
        assert svc["ports"][0].startswith("127.0.0.1:")
    assert yaml.safe_load(render.snippet(p, "kokoro-tts"))["services"]["kokoro-tts"]["image"] == "ghcr.io/owner/kokoro-tts:0.1.0"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_snippet_passes_docker_compose_config(tmp_path):
    f = tmp_path / "compose.yml"
    f.write_text(render.snippet(load_platform(), "qwen3-8b"))
    r = subprocess.run(["docker", "compose", "-f", str(f), "config", "-q"], capture_output=True, text=True,
                       env={"AI_ASSETS": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr


def test_unknown_snippet_name():
    with pytest.raises(KeyError, match="unknown service 'nope'"):
        render.snippet(load_platform(), "nope")
