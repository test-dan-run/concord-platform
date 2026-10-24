"""Nemotron 3 Diarization (Streaming Sortformer, up to 8 speakers) via NeMo, from the local .nemo file."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .turns import turns_to_tensors

MODEL_FILE = "Nemotron-3-Diarization.nemo"
# "Very high latency (offline)" configuration from the model card, in 80 ms frames: best accuracy
# for whole-file requests. Lower-latency presets only matter for streaming, which OIP does not do.
OFFLINE = {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340, "chunk_right_context": 40, "spkcache_update_period": 300}


def parse_segment(line: str) -> tuple[float, float, int]:
    """NeMo returns 'begin end speaker_N' strings."""
    start, end, label = line.split()
    return float(start), float(end), int(label.rsplit("_", 1)[-1])


class NemotronDiarizer:
    def __init__(self, weights_dir: str | Path):
        self.path = Path(weights_dir) / MODEL_FILE
        self.model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from aisvc.gpu import limit_torch_memory
        from nemo.collections.asr.models import SortformerEncLabelModel

        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} missing; run `just fetch-weights nemotron-3-diarization`")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            limit_torch_memory(torch)
        model = SortformerEncLabelModel.restore_from(restore_path=str(self.path), map_location=device, strict=False)
        model.eval()
        for key, value in OFFLINE.items():
            setattr(model.sortformer_modules, key, value)
        model._check_streaming_parameters()
        self.model = model

    def diarize(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        import torch

        assert self.model is not None, "call load() first"
        if sample_rate != 16000:
            raise ValueError(f"SAMPLE_RATE must be 16000 (the model's rate), got {sample_rate}; resample first")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 1600:  # under 0.1 s: nothing to diarize
            return turns_to_tensors([])
        with self._lock, torch.inference_mode():
            segments = self.model.diarize(audio=[audio], batch_size=1, sample_rate=sample_rate, verbose=False)
        return turns_to_tensors(parse_segment(s) for s in segments[0])
