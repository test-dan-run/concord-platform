"""Qwen3-ForcedAligner-0.6B through the `qwen-asr` 0.0.6 package (installed without its dependencies).

The package's `Qwen3ForcedAligner.align` returns timestamps only. This follows its steps for one clip (tokenise the
transcript, put two <timestamp> slots after each word, take the argmax class of each slot x 80 ms, then the
package's monotonic fix) and also keeps the softmax probability of each slot's class: a word's score is the mean of
its two. `nagisa` (Japanese tokeniser) is imported by the package at module level but has no Python 3.12 wheel, so
a stub stands in for it; Japanese is not supported here.

Many clips align in one padded forward pass (`slot_classes_batch`); the package's own `align` batches the same way.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
MAX_S = 180.0  # qwen-asr's MAX_FORCE_ALIGN_INPUT_SECONDS; on the 5090 a 292 s clip aligned only to 245 s (mean score 0.15)
MIN_S = 0.1  # a shorter row is not run: no word fits, and the mel STFT fails under 201 samples
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


@dataclass
class Row:
    """One clip ready to align: 16 kHz audio, the aligner's words, and the prompt with two <timestamp> slots per word."""
    audio: np.ndarray
    words: list[str]
    prompt: str


EMPTY = ([], np.zeros((0, 2), np.float32), np.zeros(0, np.float32))


class QwenForcedAligner:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.wrapper = None
        self.device = "cpu"
        self._torch = None

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

    def row(self, audio: np.ndarray, sample_rate: int, text: str, language: str) -> Row:
        """Check and prepare one clip; raises ValueError on bad input, before any batch runs."""
        if self.wrapper is None:
            raise RuntimeError("model not loaded")
        lang, x = language_name(language), prepare(audio, sample_rate)
        words, prompt = self.words(text, lang)
        return Row(x, list(words), prompt)

    def slot_classes_batch(self, rows: list[Row]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(class index, its probability) of every <timestamp> slot of each row, from one padded forward pass.

        Each row is prepared alone, exactly as it would be on its own; prompts are padded on the left and the mel
        features on the right with their mask. The audio encoder runs each clip at its own length and the decoder
        takes positions from the attention mask, so a row gets what it gets alone (up to bf16 rounding).
        Callers serialise calls (aisvc MicroBatcher).
        """
        torch, w = self._torch, self.wrapper
        one = [w.processor(text=[r.prompt], audio=[r.audio], return_tensors="pt", padding=True) for r in rows]
        pad = w.processor.tokenizer.pad_token_id or 0  # masked out, so any id that is not a <timestamp> slot
        width, frames = max(o["input_ids"].shape[1] for o in one), max(o["input_features"].shape[-1] for o in one)
        inputs = {"input_ids": torch.full((len(one), width), pad, dtype=torch.long),
                  "attention_mask": torch.zeros((len(one), width), dtype=torch.long),
                  "input_features": torch.zeros((len(one), one[0]["input_features"].shape[1], frames)),
                  "feature_attention_mask": torch.zeros((len(one), frames), dtype=torch.long)}
        for i, o in enumerate(one):
            n, f = o["input_ids"].shape[1], o["input_features"].shape[-1]
            inputs["input_ids"][i, width - n:] = o["input_ids"][0]
            inputs["attention_mask"][i, width - n:] = o["attention_mask"][0]
            inputs["input_features"][i, :, :f] = o["input_features"][0]
            inputs["feature_attention_mask"][i, :f] = o["feature_attention_mask"][0]
        inputs = {k: v.to(w.model.device, dtype=w.model.dtype) if v.is_floating_point() else v.to(w.model.device)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            logits = w.model.thinker(**inputs).logits
            out = []
            for i in range(len(one)):
                slots = logits[i][inputs["input_ids"][i] == w.timestamp_token_id].float()
                p, idx = torch.softmax(slots, dim=-1).max(dim=-1)
                out.append((idx.cpu().numpy(), p.cpu().numpy()))
        return out

    def result(self, row: Row, idx: np.ndarray, prob: np.ndarray):
        """(words, times [n, 2] in seconds, scores [n]) from the slot classes: class x 80 ms, then the package's
        monotonic fix."""
        if len(idx) != 2 * len(row.words):
            raise RuntimeError(f"{len(idx)} timestamp slots for {len(row.words)} words")
        fixed = self.wrapper.aligner_processor.fix_timestamp(idx * self.wrapper.timestamp_segment_time)
        times = np.asarray(fixed, np.float32).reshape(-1, 2) / 1000.0
        return list(row.words), times, prob.reshape(-1, 2).mean(axis=1).astype(np.float32)

    def align_rows(self, rows: list[Row], run=None) -> list[tuple]:
        """Results of many rows; `run` runs the slot classes of rows with words and at least MIN_S of audio (default:
        one batch of them all). Other rows get EMPTY."""
        runs = [bool(r.words) and len(r.audio) >= int(MIN_S * SR) for r in rows]
        todo = [r for r, ok in zip(rows, runs) if ok]
        classes = iter((run or self.slot_classes_batch)(todo) if todo else [])
        return [self.result(r, *next(classes)) if ok else EMPTY for r, ok in zip(rows, runs)]

    def align(self, audio: np.ndarray, sample_rate: int, text: str, language: str):
        """One clip: (words, times, scores)."""
        return self.align_rows([self.row(audio, sample_rate, text, language)])[0]
