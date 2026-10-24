"""ARK-ASR-0.6B on transformers (the model's own remote code), following the authors' inference script
(AutoArk/open-audio-opd @012947b, scripts/infer/ark_asr_transformers.py): its fixed prompt, greedy decoding that stops
at any turn marker, the logits processor that blocks ids >= 151670 and the control tokens, and its text cleanup.
Audio is cut into 30 s non-overlapping chunks (the card's example caps one pass at 30 s; the script allows 40 s).
"""

from __future__ import annotations

import io
import re
import threading
from pathlib import Path

import numpy as np

SR = 16000
MAX_CHUNK_S = 30.0
PROMPT = "Please transcribe this audio."
MAX_NEW_TOKENS = 256
BLOCK_FROM_ID = 151670  # the script's --asr_block_token_id_from: ids above the text vocabulary (codec tokens)

# from ark_asr_transformers.py
SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")


def decode_audio(data: bytes) -> np.ndarray:
    """Any libsndfile format -> mono float32 at 16 kHz. Raises ValueError on undecodable input."""
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises several types
        raise ValueError(f"could not decode audio file: {exc}") from None
    x = x.mean(axis=1)
    if len(x) == 0:
        raise ValueError("audio file has no samples")
    if sr != SR:
        import soxr

        x = soxr.resample(x, sr, SR).astype(np.float32)
    return x.astype(np.float32)


def chunks(x: np.ndarray, max_s: float = MAX_CHUNK_S) -> list[np.ndarray]:
    n = int(max_s * SR)
    return [x] if len(x) <= n else [x[s:s + n] for s in range(0, len(x), n)]


def normalize_prediction_text(text: str) -> str:
    """The script's cleanup: cut at the first turn marker, drop codec tokens, collapse spaces, strip leading noise."""
    if not text:
        return ""
    cut = min([i for i in (text.find(m) for m in TURN_END_MARKERS) if i != -1], default=len(text))
    text = text[:cut].strip()
    if "<|text|>" in text:
        text = text.split("<|text|>", 1)[1]
    text = SPECIAL_TOKEN_PATTERN.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def _ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value if v is not None]
    return [int(value)]


def eos_token_ids(tokenizer) -> list[int]:
    ids = _ids(tokenizer.eos_token_id)
    for marker in TURN_END_MARKERS:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return list(dict.fromkeys(ids))


def blocked_token_ids(tokenizer, keep: set[int], block_from_id: int = BLOCK_FROM_ID) -> list[int]:
    """Special ids and control-looking added tokens below block_from_id, except the end-of-sequence ids."""
    blocked = set(_ids(getattr(tokenizer, "all_special_ids", [])))
    for token_id, meta in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).items():
        content = getattr(meta, "content", None) or (meta.get("content") if isinstance(meta, dict) else None)
        if int(token_id) < block_from_id and content and CONTROL_TOKEN_PATTERN.match(content):
            blocked.add(int(token_id))
    return sorted(blocked - keep)


class BlockTokenIds:
    """Logits processor (any callable works in a LogitsProcessorList): mask ids >= block_from_id and the listed ids."""

    def __init__(self, block_from_id: int, token_ids: list[int]):
        self.block_from_id = block_from_id
        self.token_ids = token_ids

    def __call__(self, input_ids, scores):
        vocab = scores.shape[-1]
        if self.block_from_id < vocab:
            scores[:, self.block_from_id:] = -float("inf")
        valid = [i for i in self.token_ids if 0 <= i < vocab]
        if valid:
            scores[:, valid] = -float("inf")
        return scores


class ArkAsr:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = None
        self.generate_kwargs: dict = {}
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, LogitsProcessorList

        if not (self.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"weights missing in {self.weights_dir}; run `just fetch-weights ark-asr-0-6b`")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._torch = torch
        path = str(self.weights_dir)
        # transformers warns that the tokenizer needs fix_mistral_regex=True. The script passes it, but
        # ArkasrProcessor.from_pretrained drops it, so the script's prompts are tokenized without it too: keep it off.
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                                          dtype=self.dtype, attn_implementation="sdpa")
        self.model = self.model.to(self.device).eval()
        tok = self.processor.tokenizer
        eos = eos_token_ids(tok)
        keep = set(_ids(tok.eos_token_id)) | set(_ids(self.model.config.eos_token_id)) \
            | set(_ids(self.model.generation_config.eos_token_id))
        self.generate_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "repetition_penalty": 1.0,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else eos[0], "eos_token_id": eos,
            "logits_processor": LogitsProcessorList([BlockTokenIds(BLOCK_FROM_ID, blocked_token_ids(tok, keep))]),
        }

    def transcribe_chunk(self, x: np.ndarray) -> str:
        torch = self._torch
        conversation = [{"role": "user", "content": [{"type": "audio", "array": x}, {"type": "text", "text": PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt", sampling_rate=SR, audio_padding="longest",
            text_kwargs={"padding": "longest"}, audio_max_length=int(MAX_CHUNK_S * SR))
        inputs = {k: v.to(self.device, dtype=self.dtype) if k == "audios" else v.to(self.device)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.generate_kwargs)
        new = out[0, inputs["input_ids"].shape[1]:]
        return normalize_prediction_text(self.processor.tokenizer.decode(new, skip_special_tokens=False))

    def transcribe(self, data: bytes) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        x = decode_audio(data)
        with self._lock:
            parts = [self.transcribe_chunk(c) for c in chunks(x)]
        return " ".join(p for p in parts if p), round(len(x) / SR, 3)
