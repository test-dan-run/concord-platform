"""Size a model for an engine manifest: checkpoint GB, KV cache per token, and a gpu.util suggestion.

    uv run --quiet python <this file> SOURCE [--max-model-len N] [--util U] [--overhead-gb G]
                                             [--kv-cache-dtype auto|fp8] [--revision REV]

SOURCE is a local directory, a manifest-style /models/<dir> path (resolved against AI_ASSETS), or a
Hugging Face repo id (metadata only: config.json and file sizes are read, no weights are downloaded).
Run from platform/ so httpx and pyyaml come from the platform venv.

The estimate is weights + KV(max-model-len) + overhead (activations, CUDA graphs, allocator slack).
vLLM preallocates exactly util * GPU memory, so a util that is too low fails at startup with "no
memory for cache blocks" rather than at request time. Treat the number as a starting point and
record the arithmetic in the manifest comment, like the existing manifests do.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import httpx
import yaml

PLATFORM = Path(__file__).resolve().parents[4]  # the skill lives in <platform>/.claude/skills/register-model/scripts
GIB = 1024**3
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")
NO_KV_LAYER_TYPES = {"linear_attention", "mamba", "mamba2", "conv", "ssm"}
POOLING_ARCH_HINTS = ("Model", "ForSequenceClassification", "ForMaskedLM", "ForTokenClassification")


# ----------------------------------------------------------------------------- platform settings

def platform_settings() -> tuple[float, Path]:
    """(gpu_total_gb, assets dir) from config.yaml, with AI_ASSETS from the env or platform/.env."""
    cfg = yaml.safe_load((PLATFORM / "config.yaml").read_text()) or {}
    assets = os.environ.get("AI_ASSETS")
    env_file = PLATFORM / ".env"
    if not assets and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "AI_ASSETS" and value.strip():
                assets = value.strip().strip("'\"")
    return float(cfg.get("gpu_total_gb", 32)), Path(os.path.expanduser(assets or cfg.get("assets_dir", "~/ai-assets")))


# ----------------------------------------------------------------------------- sources

def weight_files(files: dict[str, int]) -> dict[str, int]:
    """Top-level checkpoint files the engine would load (safetensors preferred, one layout only)."""
    top = {f: n for f, n in files.items() if "/" not in f and f.endswith(WEIGHT_EXTS)}
    st = {f: n for f, n in top.items() if f.endswith(".safetensors")}
    if st:
        # Mistral-style repos ship both consolidated*.safetensors and model-*.safetensors; vLLM loads one.
        hf = {f: n for f, n in st.items() if not f.startswith("consolidated")}
        return hf or st
    return top


def from_local(path: Path) -> tuple[dict, dict[str, int], str]:
    cfg_file = path / "config.json"
    if not cfg_file.is_file():
        raise SystemExit(f"{path} has no config.json (not an HF-format checkpoint; an engine probably cannot load it)")
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
    source = (path / "SOURCE.txt").read_text().strip() if (path / "SOURCE.txt").is_file() else "no SOURCE.txt"
    return json.loads(cfg_file.read_text()), files, source


def from_hub(repo: str, revision: str) -> tuple[dict, dict[str, int], str]:
    headers = {"Authorization": f"Bearer {t}"} if (t := os.environ.get("HF_TOKEN")) else {}
    with httpx.Client(base_url="https://huggingface.co", headers=headers, follow_redirects=True, timeout=30) as c:
        r = c.get(f"/api/models/{repo}/revision/{revision}", params={"blobs": "true"})
        if r.status_code in (401, 403):
            raise SystemExit(f"{repo} is gated or private: accept its license on huggingface.co and set HF_TOKEN")
        if r.status_code == 404:
            raise SystemExit(f"{repo}@{revision} not found on huggingface.co")
        r.raise_for_status()
        info = r.json()
        files = {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}
        if "config.json" not in files:
            raise SystemExit(f"{repo} has no config.json (GGUF/ONNX/other format?); files: {sorted(files)[:15]}")
        r = c.get(f"/{repo}/resolve/{info['sha']}/config.json")
        if r.status_code in (401, 403):
            raise SystemExit(f"{repo} is gated (gated={info.get('gated')}): accept its license on huggingface.co "
                             "and set HF_TOKEN, then rerun")
        cfg = r.raise_for_status().json()
    gated = f", gated={info['gated']}" if info.get("gated") else ""
    return cfg, files, f"repo: {repo}\nrevision: {info['sha']}{gated}"


# ----------------------------------------------------------------------------- KV cache

def kv_bytes_per_token(cfg: dict, kv_dtype_bytes: int) -> tuple[int | None, str]:
    """Bytes of KV cache per token across all layers that keep one, plus a note on how it was derived."""
    t = cfg.get("text_config") or cfg
    if "kv_lora_rank" in t:  # MLA (DeepSeek-style): one compressed latent + rope key per layer
        layers = t["num_hidden_layers"]
        per = (t["kv_lora_rank"] + t.get("qk_rope_head_dim", 64)) * kv_dtype_bytes
        return layers * per, f"MLA: {layers} layers x (kv_lora_rank {t['kv_lora_rank']} + rope {t.get('qk_rope_head_dim', 64)})"

    layers = t.get("num_hidden_layers") or t.get("decoder_layers") or t.get("n_layer")
    heads = t.get("num_attention_heads") or t.get("decoder_attention_heads") or t.get("n_head")
    kv_heads = t.get("num_key_value_heads") or heads
    hidden = t.get("hidden_size") or t.get("d_model") or t.get("n_embd")
    head_dim = t.get("head_dim") or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        return None, "could not read layers / kv heads / head_dim from config.json"

    note = ""
    layer_types = t.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        kv_layers = [lt for lt in layer_types if lt not in NO_KV_LAYER_TYPES]
        skipped = len(layer_types) - len(kv_layers)
        if skipped:
            note = f"; hybrid: {len(kv_layers)} of {len(layer_types)} layers keep a KV cache (others hold fixed-size state)"
        if any("sliding" in lt for lt in kv_layers):
            note += "; sliding-window layers cache only the window, so this is an upper bound"
        layers = len(kv_layers)
    if cfg.get("is_encoder_decoder"):
        note += "; encoder-decoder: decoder self-attention only (cross-attention cache is per request, small)"
    per = 2 * layers * kv_heads * head_dim * kv_dtype_bytes
    return per, f"2 (K,V) x {layers} layers x {kv_heads} kv heads x {head_dim} head_dim x {kv_dtype_bytes} B{note}"


def describe_quant(cfg: dict) -> str:
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if not q:
        return f"none (dtype {cfg.get('torch_dtype') or cfg.get('dtype') or (cfg.get('text_config') or {}).get('dtype') or '?'})"
    keys = ("quant_method", "bits", "fmt", "weight_block_size", "group_size")
    return ", ".join(f"{k}={q[k]}" for k in keys if q.get(k) is not None)


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="local dir, /models/<dir>, or HF repo id")
    ap.add_argument("--revision", default="main", help="HF revision (hub sources only)")
    ap.add_argument("--max-model-len", type=int, help="context to budget KV for (default: config max, capped at 32768)")
    ap.add_argument("--util", type=float, help="instead: how many KV tokens fit at this gpu.util")
    ap.add_argument("--overhead-gb", type=float, default=1.5,
                    help="activations + CUDA graphs + slack; ~1-2 GB for <=14B text models, add ~2 GB for VLMs (default 1.5)")
    ap.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    args = ap.parse_args(argv)

    total_gb, assets = platform_settings()
    src = args.source
    local = Path(os.path.expanduser(src))
    if src.startswith("/models/"):
        local = assets / src.removeprefix("/models/")
    if local.is_dir():
        cfg, files, provenance = from_local(local)
        where = str(local)
    elif local.is_absolute() or src.startswith((".", "~")):
        raise SystemExit(f"{local} does not exist")
    else:
        cfg, files, provenance = from_hub(src, args.revision)
        where = f"https://huggingface.co/{src}"

    weights = weight_files(files)
    weights_gb = sum(weights.values()) / GIB
    archs = cfg.get("architectures") or []
    t = cfg.get("text_config") or cfg
    max_pos = t.get("max_position_embeddings") or t.get("max_target_positions") or t.get("n_positions")

    print(f"source         {where}")
    for line in provenance.splitlines():
        print(f"               {line}")
    print(f"architectures  {', '.join(archs) or '?'}  (model_type {cfg.get('model_type')})")
    print(f"quantization   {describe_quant(cfg)}")
    print(f"multimodal     {'yes (vision_config present: add encoder-cache headroom)' if 'vision_config' in cfg else 'no'}")
    print(f"max context    {max_pos}")
    print(f"weights        {weights_gb:.2f} GB in {len(weights)} file(s): "
          f"{', '.join(sorted(weights)[:3])}{' ...' if len(weights) > 3 else ''}")
    if not weights:
        print("               no .safetensors/.bin at the top level: check the repo layout before writing a manifest")
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or t.get("dtype")
    if dtype == "float32" and not (cfg.get("quantization_config") or t.get("quantization_config")):
        weights_gb /= 2
        print(f"               fp32 checkpoint: vLLM loads it as fp16/bf16, so budget {weights_gb:.2f} GB")

    pooling = any(a.endswith(POOLING_ARCH_HINTS) for a in archs) and not any("CausalLM" in a or "Generation" in a for a in archs)
    kv_dtype_bytes = 1 if args.kv_cache_dtype == "fp8" else 2
    per_tok, how = kv_bytes_per_token(cfg, kv_dtype_bytes)
    if pooling:
        print("kv cache       pooling/encoder model (embeddings, rerank): no decode KV cache; budget weights + overhead")
        per_tok = 0
    elif per_tok is None:
        print(f"kv cache       unknown: {how}")
        return 0
    else:
        print(f"kv per token   {per_tok / 1024:.0f} KB  = {how}")

    if args.util is not None:
        budget = args.util * total_gb - weights_gb - args.overhead_gb
        print(f"\nat util {args.util:g} ({args.util * total_gb:.1f} GB): {budget:.2f} GB left for KV"
              + (f" ~= {int(budget * GIB / per_tok):,} tokens" if per_tok and budget > 0 else ""))
        if budget <= 0:
            print("  too small: weights + overhead alone exceed this budget")
        return 0

    max_len = args.max_model_len or min(max_pos or 8192, 32768)
    kv_gb = per_tok * max_len / GIB
    need = weights_gb + kv_gb + args.overhead_gb
    util = math.ceil(need / total_gb * 100) / 100
    print(f"\nfor --max-model-len={max_len}: weights {weights_gb:.1f} + KV {kv_gb:.2f} + overhead {args.overhead_gb:g} "
          f"= {need:.1f} GB  ->  gpu.util {util:.2f} ({util * total_gb:.1f} GB of {total_gb:g})")
    if max_pos and max_len < max_pos:
        print(f"  capped below the native {max_pos}: put --max-model-len={max_len} in args with a comment")
    if (weights_gb + args.overhead_gb) / total_gb > 0.93:
        print("  weights alone exceed the platform util_limit (0.93): needs a smaller checkpoint (AWQ/GPTQ 4-bit, FP8)")
    elif util > 0.93:
        print("  over the platform util_limit (0.93): shorten --max-model-len or add --kv-cache-dtype=fp8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
