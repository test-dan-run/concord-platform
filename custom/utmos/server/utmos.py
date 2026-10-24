"""UTMOS22 strong learner: a predicted MOS (about 1-5) for one clip of speech.

The model code ships with the weights (transformers remote code, `modeling_utmos22.py`), so it is loaded from
the mounted weights dir with `trust_remote_code`. It resamples to 16 kHz itself and pads clips shorter
than 400 samples.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


class Utmos:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self._torch = None
        self._lock = threading.Lock()

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
        if self.model is None:
            raise RuntimeError("model not loaded")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            raise ValueError("empty clip")
        torch = self._torch
        with self._lock, torch.inference_mode():
            out = self.model(torch.from_numpy(x), sampling_rate=int(sample_rate))
        return float(out.scores[0])
