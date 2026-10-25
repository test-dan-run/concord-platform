"""ReDimNet2-B6 speaker embeddings (192-d) for a batch of clips.

The model has no padding mask (its front end normalises each whole signal, and its 1-D blocks attend over time),
so only clips of equal length after `prepare` share a forward pass: every clip of MAX_S or more, every clip of
MIN_S or less, and repeated lengths. Other clips run one by one, which gives the same embeddings as before.

The model code is fetched with the weights into `<weights>/code/redimnet2` (PalabraAI/redimnet2 @c5bbe0b) and
imported from there; the checkpoint holds its own `model_config`. Input is 16 kHz audio in [-1, 1] (the model's
front end normalises the signal); other rates are resampled. Clips shorter than MIN_S are repeated to that length,
so the pooling has enough frames.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CHECKPOINT = "b6-vb2+vox2+cnc2_v0-lm.pt"
SR = 16000
MIN_S = 0.5
MAX_S = 20.0  # longer clips are cut: 60 s ran out of the 2 GB budget on the 5090, and 20 s identifies a speaker well


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


class ReDimNet2Embedder:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.device = "cpu"
        self._torch = None

    def load(self) -> None:
        import torch

        ckpt = self.weights_dir / CHECKPOINT
        if not ckpt.is_file():
            raise FileNotFoundError(f"{ckpt} missing; run `just fetch-weights redimnet2-sv`")
        sys.path.insert(0, str(self.weights_dir / "code"))
        from redimnet2.redimnet2 import ReDimNet2Wrap

        full = torch.load(ckpt, map_location="cpu", weights_only=False)  # holds model_config next to the weights
        model = ReDimNet2Wrap(**full["model_config"])
        res = model.load_state_dict(full["state_dict"])
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
        if torch.cuda.is_available() and os.environ.get("SV_DEVICE", "cuda") == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
            self.device = "cuda"
        self._torch = torch
        self.model = model.eval().to(self.device)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.embed_batch([prepare(audio, sample_rate)])[0]

    def embed_batch(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """Embeddings of prepared clips (`prepare`): one forward pass per group of equal length.
        Callers serialise calls (aisvc.batch.MicroBatcher)."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        torch = self._torch
        groups: dict[int, list[int]] = {}
        for i, c in enumerate(clips):
            groups.setdefault(len(c), []).append(i)
        out: list[np.ndarray | None] = [None] * len(clips)
        with torch.inference_mode():
            for idx in groups.values():
                x = torch.from_numpy(np.stack([clips[i] for i in idx])).to(self.device)
                e = torch.nn.functional.normalize(self.model(x).float(), dim=-1).cpu().numpy()
                for j, i in enumerate(idx):
                    out[i] = e[j]
        return out
