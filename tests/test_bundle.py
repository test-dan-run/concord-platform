"""bundle.py: weights, platform tree, directory layout and dry-run report, on a fake AI_ASSETS with
Docker mocked out (no images are saved, no real weights are read)."""

import dataclasses
import json
import tarfile

import pytest
import yaml

from tools import bundle, render
from tools.schema import load_platform

INDEX = {"weight_map": {"a.w": "model-00001-of-00002.safetensors", "b.w": "model-00002-of-00002.safetensors"}}


@pytest.fixture
def assets(tmp_path, monkeypatch):
    a = tmp_path / "ai-assets"
    (a / "custom-models" / "silero-vad" / "v6.2.3").mkdir(parents=True)
    (a / "custom-models" / "silero-vad" / "v6.2.3" / "silero_vad.onnx").write_bytes(b"onnx" * 100)
    (a / "whisper-large-v3").mkdir()
    (a / "whisper-large-v3" / "model.safetensors").write_bytes(b"w" * 3000)
    (a / "whisper-large-v3" / "config.json").write_text("{}")
    asr = a / "qwen3-asr-1.7b"  # a sharded local copy
    asr.mkdir()
    (asr / "config.json").write_text("{}")
    (asr / "model.safetensors.index.json").write_text(json.dumps(INDEX))
    (asr / "model-00001-of-00002.safetensors").write_bytes(b"1" * 5000)
    (asr / "model-00002-of-00002.safetensors").write_bytes(b"2" * 7000)
    (a / "whisper-large-v3-turbo").mkdir()
    (a / "whisper-large-v3-turbo" / "model.safetensors").write_bytes(b"t" * 900)
    monkeypatch.setenv("AI_ASSETS", str(a))
    return a


@pytest.fixture
def docker(monkeypatch):
    """Every image present (ids and sizes made up); `docker save` writes a placeholder file."""
    saved = []
    monkeypatch.setattr(bundle, "docker_available", lambda: True)
    monkeypatch.setattr(bundle, "image_facts", lambda ref: {"id": f"sha256:{abs(hash(ref)):x}", "bytes": 1000})

    def save(refs, dest):
        saved.append(list(refs))
        dest.write_bytes(b"tar")

    monkeypatch.setattr(bundle, "save_images", save)
    return saved


def _row(rep, name):
    return next(r for r in rep["services"] if r["name"] == name)


def test_local_copy_is_copied_as_real_files(assets, docker, tmp_path):
    (assets / "qwen3-asr-1.7b" / "tokenizer.json").symlink_to(assets / "qwen3-asr-1.7b" / "config.json")
    p = load_platform()
    out = tmp_path / "kit"
    bundle.write_bundle(p, p.resolve(["qwen3-asr-1-7b"]), out, with_weights=True, log=lambda m: None)
    shard = out / "assets" / "qwen3-asr-1.7b" / "model-00002-of-00002.safetensors"
    assert shard.is_file() and shard.read_bytes() == b"2" * 7000
    assert not any(f.is_symlink() for f in out.rglob("*"))


def test_weight_sizes_are_apparent_sizes(assets, docker):
    p = load_platform()
    rep = bundle.report(p, p.resolve(["qwen3-asr-1-7b", "silero-vad", "whisper-large-v3", "whisper-large-v3-turbo"]))
    asr = _row(rep, "qwen3-asr-1-7b")
    assert asr["weights"]["kind"] == "local" and asr["weights"]["path"] == "qwen3-asr-1.7b"
    assert asr["weights_bytes"] == 2 + len(json.dumps(INDEX)) + 12000
    assert _row(rep, "silero-vad")["weights"] | {"source": None} == {
        "kind": "custom", "source": None, "path": "custom-models/silero-vad/v6.2.3", "files": 1, "bytes": 400,
        "problems": []}
    assert _row(rep, "whisper-large-v3")["weights"]["path"] == "whisper-large-v3"  # model: /models/whisper-large-v3
    assert rep["weights_bytes"] == sum(r["weights_bytes"] for r in rep["services"])
    assert rep["images"]["refs"] == ["ghcr.io/owner/silero-vad:0.1.0", "vllm/vllm-openai:v0.31.0"]
    assert rep["images"]["bytes"] == 2000 and rep["problems"] == []


def test_partly_downloaded_model_fails_the_bundle(assets, docker, tmp_path):
    (assets / "qwen3-asr-1.7b" / "model-00002-of-00002.safetensors").unlink()
    p = load_platform()
    rep = bundle.report(p, p.resolve(["qwen3-asr-1-7b"]))
    assert rep["problems"] == ["qwen3-asr-1-7b: incomplete download: model-00002-of-00002.safetensors named in "
                               "model.safetensors.index.json missing"]
    with pytest.raises(bundle.BundleError, match="incomplete download"):
        bundle.write_bundle(p, p.resolve(["qwen3-asr-1-7b"]), tmp_path / "kit", with_weights=True, log=lambda m: None)


def test_broken_link_is_a_problem(assets, docker):
    (assets / "qwen3-asr-1.7b" / "tokenizer.json").symlink_to(assets / "nowhere")
    p = load_platform()
    assert any("broken link tokenizer.json" in x for x in bundle.report(p, p.resolve(["qwen3-asr-1-7b"]))["problems"])


def test_missing_images_and_weights(assets, monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "docker_available", lambda: True)
    monkeypatch.setattr(bundle, "image_facts", lambda ref: None)
    p = load_platform()
    rep = bundle.report(p, p.resolve(["kokoro-tts", "qwen3-8b"]))
    kokoro = _row(rep, "kokoro-tts")
    assert kokoro["image_present"] is False and "`just build kokoro-tts`" in kokoro["image_problem"]
    assert "`docker pull vllm/vllm-openai:v0.31.0`" in _row(rep, "qwen3-8b")["image_problem"]
    assert any("weights not found" in x and "just fetch-weights kokoro-tts" in x for x in rep["problems"])
    assert any("local model /models/qwen3-8b-fp8 not found" in x for x in rep["problems"])
    with pytest.raises(bundle.BundleError, match="not present locally"):
        bundle.write_bundle(p, p.resolve(["text-normalizer"]), tmp_path / "a", with_images=True, log=lambda m: None)
    # without images or weights the same problems block nothing
    bundle.write_bundle(p, p.resolve(["kokoro-tts"]), tmp_path / "b", log=lambda m: None)
    assert (tmp_path / "b" / "services" / "kokoro-tts.yaml").is_file()


def test_platform_tree_holds_only_the_chosen_services_and_runs(assets, docker, tmp_path):
    p = load_platform()
    out = tmp_path / "kit"
    bundle.write_bundle(p, p.resolve(["default-vad", "whisper-large-v3"]), out, platform_tree=True,
                        log=lambda m: None)
    tree = out / "platform"
    assert sorted(f.name for f in (tree / "services").iterdir()) == ["silero-vad.yaml", "whisper-large-v3.yaml"]
    assert (tree / "custom" / "silero-vad" / "Dockerfile").is_file() and not (tree / "custom" / "kokoro-tts").exists()
    for rel in ("tools/stack.py", "engines/engines.yaml", "contracts/api_types.yaml", "config.yaml", "uv.lock",
                "pyproject.toml", "clients/aicore/pyproject.toml", "custom/_bases/bases.yaml", "ui/template.html.j2"):
        assert (tree / rel).is_file(), rel
    assert not list(tree.rglob("__pycache__")) and not (tree / ".venv").exists() and not (tree / ".env").exists()
    compose = yaml.safe_load((tree / "compose" / "generated" / "compose.yml").read_text())
    assert sorted(compose["services"]) == ["silero-vad", "whisper-large-v3"]
    eps = yaml.safe_load((tree / "clients" / "aicore" / "src" / "aicore" / "endpoints.generated.yaml").read_text())
    assert sorted(eps["services"]) == ["default-asr", "default-vad", "silero-vad", "whisper-large-v3"]
    assert (tree / "ui" / "catalog.html").is_file()
    t = load_platform(tree)
    assert [s.name for s in t.services] == ["silero-vad", "whisper-large-v3"]
    assert not (out / "compose.yml").exists() and not (out / "services").exists()  # the tree replaces them


def test_full_directory_layout(assets, docker, tmp_path):
    p = load_platform()
    out = tmp_path / "kit"
    lines = []
    rep = bundle.write_bundle(p, p.resolve(["silero-vad", "whisper-large-v3-turbo"]), out, with_images=True,
                              with_weights=True, platform_tree=True, log=lines.append)
    assert docker == [["ghcr.io/owner/silero-vad:0.1.0", "vllm/vllm-openai:v0.31.0"]]  # one docker save
    assert (out / "images" / "platform.tar").is_file()
    assert (out / "assets" / "custom-models" / "silero-vad" / "v6.2.3" / "silero_vad.onnx").read_bytes() == b"onnx" * 100
    saved = json.loads((out / "bundle.json").read_text())
    assert saved == json.loads(json.dumps(rep)) and saved["with_weights"] and saved["platform_tree"]
    assert _row(saved, "silero-vad")["image_id"].startswith("sha256:")
    assert [x for x in lines if x.startswith("==")] == ["== platform tree", "== weights", "== images", f"== done: {out}"]
    readme = (out / "README.md").read_text()
    assert "docker load -i images/platform.tar" in readme and "no internet access is needed" in readme
    with pytest.raises(bundle.BundleError, match="is not empty"):
        bundle.write_bundle(p, p.resolve(["silero-vad"]), out, log=lambda m: None)


def test_tarball_keeps_the_old_contents(assets, docker, tmp_path):
    p = load_platform()
    target = bundle.make_bundle(p, p.resolve(["silero-vad", "qwen3-8b"]), tmp_path, with_images=False,
                                log=lambda m: None)
    assert target.name == "aiplatform-silero-vad-qwen3-8b.tar.gz"
    with tarfile.open(target) as tar:
        names = sorted(tar.getnames())
    root = "aiplatform-silero-vad-qwen3-8b"
    assert names == [f"{root}/{n}" for n in ("README.md", "bundle.json", "compose.yml", "endpoints.generated.yaml",
                                             "services/qwen3-8b.yaml", "services/silero-vad.yaml",
                                             "weights/silero-vad.weights.yaml")]


def test_dry_run_json_and_budget_warning(assets, docker, capsys):
    assert bundle.main(["qwen-sea-lion-v4-5-27b-it", "gemma-sea-lion-v4-27b-it-nvfp4", "--dry-run", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert not rep["budget"]["ok"] and any("over util_limit" in w for w in rep["warnings"])
    assert bundle.main(["--all", "--dry-run", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["services"]) == len(load_platform().services)


def test_cli_rejects_json_without_dry_run_and_names_with_all():
    with pytest.raises(SystemExit):
        bundle.main(["silero-vad", "--json"])
    with pytest.raises(SystemExit):
        bundle.main(["silero-vad", "--all", "--dry-run"])


def test_compose_project_names_the_rendered_stack():
    p = load_platform()
    kit = dataclasses.replace(p, config=p.config.model_copy(update={"compose_project": "concord-kit"}))
    assert render.build_compose(kit, kit.root / "compose" / "generated")["name"] == "concord-kit"
    assert render.build_compose(p, p.root / "compose" / "generated")["name"] == "aiplatform"
    assert bundle.bundle_compose(kit, kit.resolve(["silero-vad"]))["name"] == "concord-kit"
