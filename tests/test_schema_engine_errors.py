"""An invalid engines.yaml entry is reported once, with its name once, and manifests say the engine is invalid."""

import shutil

import pytest
import yaml
from factories import base_engine_manifest

from tools.schema import PLATFORM_DIR, PlatformError, load_platform


@pytest.fixture
def root(tmp_path):
    """A platform root with config, contracts and bases copied, a broken vllm engine and one vllm manifest."""
    for rel in ("config.yaml", "contracts/api_types.yaml", "custom/_bases/bases.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PLATFORM_DIR / rel, tmp_path / rel)
    (tmp_path / "engines").mkdir()
    engine = {"image": "vllm/vllm-openai:latest", "contract": "openai", "internal_port": 8000,
              "health_path": "/health", "command": []}
    (tmp_path / "engines" / "engines.yaml").write_text(yaml.safe_dump({"vllm": engine}))
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "test-llm.yaml").write_text(yaml.safe_dump(base_engine_manifest()))
    return tmp_path


def test_an_invalid_engine_is_named_once_and_manifests_say_it_is_invalid(root):
    with pytest.raises(PlatformError) as exc:
        load_platform(root)
    errors = exc.value.errors
    assert "engines.yaml: vllm: image: engine image 'vllm/vllm-openai:latest' uses :latest; pin an exact tag" in errors
    assert "test-llm.yaml: (manifest): engine 'vllm' is invalid in engines.yaml" in errors
    assert not any("is not defined" in e for e in errors)
