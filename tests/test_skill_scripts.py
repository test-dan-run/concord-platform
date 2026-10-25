"""The register-model skill's helper scripts run against the real platform and agree with tools/schema.py."""

import importlib.util

from tools.schema import PLATFORM_DIR, load_platform

SCRIPTS = PLATFORM_DIR / ".claude" / "skills" / "register-model" / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"skill_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_state_prints_the_next_free_port_for_an_api(capsys):
    platform_state = load_script("platform_state")
    assert platform_state.main(["--api", "infer"]) == 0
    out = capsys.readouterr().out
    used = {s.port for s in load_platform().services}
    free = next(port for port in range(8401, 8500) if port not in used)
    assert f"--api infer: contract oip-v2, band 84xx (other models), next free port {free}" in out


def test_platform_state_uses_the_schema_port_bands(capsys):
    platform_state = load_script("platform_state")
    assert platform_state.main(["--api", "text.itn", "--kind", "utility"]) == 0
    assert "band 85xx (utilities)" in capsys.readouterr().out


def test_size_model_reads_util_limit_from_config_yaml():
    size_model = load_script("size_model")
    total_gb, util_limit, _ = size_model.platform_settings()
    config = load_platform().config
    assert (total_gb, util_limit) == (config.gpu_total_gb, config.util_limit)
