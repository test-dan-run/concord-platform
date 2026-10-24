"""WavLM-Large + ECAPA-TDNN speaker embeddings (256-d) for one clip: UniSpeech's speaker-verification model, the
SIM model of Seed-TTS-eval.

The model code ships with the weights (transformers remote code, `modeling_wavlm_speaker_verification.py`), so it
is loaded from the mounted weights dir with `trust_remote_code`. Input is resampled to 16 kHz here; clips shorter
than MIN_S are repeated to that length, so the pooling has enough frames; longer than MAX_S are cut, since WavLM's
attention grows with the square of the length (60 s ran out of the 3 GB budget on an RTX 5090) and 20 s is more
than enough to identify a speaker.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

SR = 16000
MIN_S = 0.5
MAX_S = 20.0


def prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz, repeated up to MIN_S and cut at MAX_S."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        raise ValueError("empty clip")
    if sample_rate != SR:
        import soxr

        x = soxr.resample(x, sample_rate, SR).astype(np.float32)
    n_min = int(MIN_S * SR)
    if len(x) < n_min:
        x = np.tile(x, -(-n_min // len(x)))[:n_min]
    return x[: int(MAX_S * SR)]


class WavLMEmbedder:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.device = "cpu"
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from transformers import AutoModel

        if not (self.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"{self.weights_dir / 'model.safetensors'} missing; run `just fetch-weights wavlm-sv`")
        if torch.cuda.is_available() and os.environ.get("SV_DEVICE", "cuda") == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
            self.device = "cuda"
        self._torch = torch
        self.model = AutoModel.from_pretrained(str(self.weights_dir), trust_remote_code=True,
                                               local_files_only=True).eval().to(self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not loaded")
        torch = self._torch
        x = torch.from_numpy(prepare(audio, sample_rate))[None, :].to(self.device)
        with self._lock, torch.inference_mode():
            e = self.model(x, sampling_rate=SR, input_lengths=torch.tensor([x.shape[1]], device=self.device)).embeddings.float()
            e = torch.nn.functional.normalize(e, dim=-1)
        return e[0].cpu().numpy()
