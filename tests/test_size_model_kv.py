"""size_model.py counts the KV cache of encoder-decoder, thinker_config and sliding-window models from config.json."""

import importlib.util
import json

import pytest

from tools.schema import PLATFORM_DIR

SCRIPT = PLATFORM_DIR / ".claude" / "skills" / "register-model" / "scripts" / "size_model.py"
GIB = 1024**3

WHISPER_TURBO = {  # the fields of whisper-large-v3-turbo's config.json that matter: 32 encoder, 4 decoder layers
    "architectures": ["WhisperForConditionalGeneration"], "is_encoder_decoder": True, "num_hidden_layers": 32,
    "encoder_layers": 32, "decoder_layers": 4, "encoder_attention_heads": 20, "decoder_attention_heads": 20,
    "d_model": 1280, "max_target_positions": 448, "torch_dtype": "float16",
}
QWEN3_ASR = {  # Qwen3-ASR keeps its language model's config under thinker_config.text_config
    "architectures": ["Qwen3ASRForConditionalGeneration"],
    "thinker_config": {"text_config": {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                                       "head_dim": 128, "hidden_size": 2048, "max_position_embeddings": 65536}},
}
GEMMA3_27B = {  # 10 full-attention layers and 52 sliding-window layers of 1024 tokens
    "architectures": ["Gemma3ForConditionalGeneration"],
    "text_config": {"num_hidden_layers": 62, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128,
                    "hidden_size": 5376, "max_position_embeddings": 131072, "sliding_window": 1024,
                    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10 + ["sliding_attention"] * 2},
}


@pytest.fixture(scope="module")
def size_model():
    spec = importlib.util.spec_from_file_location("skill_size_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(size_model, tmp_path, capsys, cfg: dict, *args: str) -> str:
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    assert size_model.main([str(tmp_path), *args]) == 0
    return capsys.readouterr().out


def test_an_encoder_decoder_model_caches_only_its_decoder_layers(size_model, tmp_path, capsys):
    out = run(size_model, tmp_path, capsys, WHISPER_TURBO)
    assert "kv per token   20 KB  = 2 (K,V) x 4 layers x 20 kv heads x 64 head_dim x 2 B" in out


def test_the_text_config_under_thinker_config_is_read(size_model, tmp_path, capsys):
    out = run(size_model, tmp_path, capsys, QWEN3_ASR, "--max-model-len", "8192")
    assert "kv per token   112 KB  = 2 (K,V) x 28 layers x 8 kv heads x 128 head_dim x 2 B" in out
    assert "max context    65536" in out


def test_sliding_window_layers_cache_at_most_the_window(size_model, tmp_path, capsys):
    out = run(size_model, tmp_path, capsys, GEMMA3_27B, "--max-model-len", "32768")
    per_layer = 2 * 16 * 128 * 2
    kv_gib = (10 * per_layer * 32768 + 52 * per_layer * 1024) / GIB
    assert f"KV {kv_gib:.2f} + overhead" in out and kv_gib == pytest.approx(2.91, abs=0.01)
    out = run(size_model, tmp_path, capsys, GEMMA3_27B, "--max-model-len", "512")
    assert f"KV {62 * per_layer * 512 / GIB:.2f} + overhead" in out  # shorter than the window: every layer counts
