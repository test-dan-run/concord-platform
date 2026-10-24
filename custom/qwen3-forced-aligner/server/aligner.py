"""Qwen3-ForcedAligner-0.6B through the `qwen-asr` 0.0.6 package (installed without its dependencies).

The package's `Qwen3ForcedAligner.align` returns timestamps only. This follows its steps for one clip (tokenise the
transcript, put two <timestamp> slots after each word, take the argmax class of each slot x 80 ms, then the
package's monotonic fix) and also keeps the softmax probability of each slot's class: a word's score is the mean of
its two. `nagisa` (Japanese tokeniser) is imported by the package at module level but has no Python 3.12 wheel, so
a stub stands in for it; Japanese is not supported here.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np

SR = 16000
MAX_S = 180.0  # qwen-asr's MAX_FORCE_ALIGN_INPUT_SECONDS; on the 5090 a 292 s clip aligned only to 245 s (mean score 0.15)
LANGUAGES = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French", "de": "German", "it": "Italian",
             "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "es": "Spanish"}


def language_name(lang: str) -> str:
    """ISO code or name -> the package's language name. Japanese is refused (its tokeniser is not installed)."""
    key = lang.strip().lower()
    by_name = {v.lower(): v for v in LANGUAGES.values()}
    if key in LANGUAGES:
        return LANGUAGES[key]
    if key in by_name:
        return by_name[key]
    raise ValueError(f"LANGUAGE {lang!r} is not supported; use one of {sorted(LANGUAGES)} (or their names)")


def prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate < 8000:
        raise ValueError(f"SAMPLE_RATE must be at least 8000, got {sample_rate}")
    if len(x) == 0:
        raise ValueError("AUDIO is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("AUDIO has NaN or inf values")
    if len(x) / sample_rate > MAX_S:
        raise ValueError(f"AUDIO is {len(x) / sample_rate:.1f} s; at most {MAX_S:.0f} s per request")
    if sample_rate != SR:
        import soxr

        x = soxr.resample(x, sample_rate, SR).astype(np.float32)
    return x


def _stub_nagisa() -> None:
    if "nagisa" not in sys.modules:
        stub = types.ModuleType("nagisa")

        def tagging(_text):
            raise ValueError("Japanese alignment is not supported by this service")

        stub.tagging = tagging
        sys.modules["nagisa"] = stub


class QwenForcedAligner:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.wrapper = None
        self.device = "cpu"
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch

        if not (self.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"{self.weights_dir / 'model.safetensors'} missing; run `just fetch-weights qwen3-forced-aligner`")
        _stub_nagisa()
        from qwen_asr import Qwen3ForcedAligner

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._torch = torch
        self.wrapper = Qwen3ForcedAligner.from_pretrained(str(self.weights_dir), dtype=dtype, local_files_only=True)
        self.wrapper.model.to(self.device).eval()

    def words(self, text: str, language: str) -> tuple[list[str], str]:
        return self.wrapper.aligner_processor.encode_timestamp(text, language)

    def slot_classes(self, audio: np.ndarray, aligner_text: str) -> tuple[np.ndarray, np.ndarray]:
        """(class index, its probability) for every <timestamp> slot, in order."""
        torch, w = self._torch, self.wrapper
        inputs = w.processor(text=[aligner_text], audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(w.model.device).to(w.model.dtype)
        with torch.inference_mode():
            logits = w.model.thinker(**inputs).logits[0].float()
        slots = inputs["input_ids"][0] == w.timestamp_token_id
        probs = torch.softmax(logits[slots], dim=-1)
        p, idx = probs.max(dim=-1)
        return idx.cpu().numpy(), p.cpu().numpy()

    def align(self, audio: np.ndarray, sample_rate: int, text: str, language: str):
        if self.wrapper is None:
            raise RuntimeError("model not loaded")
        lang = language_name(language)
        x = prepare(audio, sample_rate)
        with self._lock:
            word_list, aligner_text = self.words(text, lang)
            if not word_list:
                return [], np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
            idx, prob = self.slot_classes(x, aligner_text)
        if len(idx) != 2 * len(word_list):
            raise RuntimeError(f"{len(idx)} timestamp slots for {len(word_list)} words")
        ms = idx * self.wrapper.timestamp_segment_time
        fixed = self.wrapper.aligner_processor.fix_timestamp(ms)
        times = np.asarray(fixed, np.float32).reshape(-1, 2) / 1000.0
        scores = prob.reshape(-1, 2).mean(axis=1).astype(np.float32)
        return list(word_list), times, scores
