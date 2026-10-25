"""UTMOS22 strong learner: a predicted MOS (about 1-5) per clip of speech, a batch of clips at a time.

The model code ships with the weights (transformers remote code, `modeling_utmos22.py`), so it is loaded from
the mounted weights dir with `trust_remote_code`. It resamples to 16 kHz itself, pads clips shorter
than 400 samples, and pads a batch with a mask, so one call scores many clips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class Utmos:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel

        if not (self.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"{self.weights_dir / 'model.safetensors'} missing; run `just fetch-weights utmos`")
        torch.set_num_threads(int(os.environ.get("UTMOS_THREADS", "4")))
        self._torch = torch
        self.model = AutoModel.from_pretrained(str(self.weights_dir), trust_remote_code=True,
                                               local_files_only=True).eval()

    def score(self, audio: np.ndarray, sample_rate: int) -> float:
        return self.score_batch([(np.asarray(audio, dtype=np.float32).reshape(-1), int(sample_rate))])[0]

    def score_batch(self, clips: list[tuple[np.ndarray, int]]) -> list[float]:
        """MOS of each (clip, sample rate), one forward pass per sample rate. Callers serialise calls
        (aisvc.batch.MicroBatcher)."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        if any(len(x) == 0 for x, _ in clips):
            raise ValueError("empty clip")
        torch = self._torch
        out: list[float | None] = [None] * len(clips)
        by_rate: dict[int, list[int]] = {}
        for i, (_, sr) in enumerate(clips):
            by_rate.setdefault(sr, []).append(i)
        with torch.inference_mode():
            for sr, idx in by_rate.items():
                scores = self.model([torch.from_numpy(clips[i][0]) for i in idx], sampling_rate=sr).scores
                for j, i in enumerate(idx):
                    out[i] = float(scores[j])
        return out
