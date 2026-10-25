"""Nemotron 3 Diarization (Streaming Sortformer, up to 8 speakers) via NeMo, from the local .nemo file.

Batches go through NeMo's own batch path: `diarize` pads the clips with zeros and passes each clip's length,
the encoder and the speaker predictions are masked by those lengths, and the speaker cache and FIFO of a row
only take in padding after that row has ended. So a row gets the same segments in a batch as alone, apart
from rounding (tests/test_nemotron_3_diarization.py checks it on the real model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .turns import turns_to_tensors

MODEL_FILE = "Nemotron-3-Diarization.nemo"
SR = 16000
MIN_SAMPLES = SR // 10  # under 0.1 s: nothing to diarize
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
        return self.diarize_batch([prepare(audio, sample_rate)])[0]

    def diarize_batch(self, clips: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Tensors of each prepared clip (`prepare`), all in one padded NeMo batch. Callers serialise calls
        (aisvc.batch.MicroBatcher)."""
        import torch

        assert self.model is not None, "call load() first"
        out = [turns_to_tensors([]) for _ in clips]
        keep = [i for i, x in enumerate(clips) if len(x) >= MIN_SAMPLES]
        if keep:
            with torch.inference_mode():
                segments = self.model.diarize(audio=[clips[i] for i in keep], batch_size=len(keep), sample_rate=SR,
                                              verbose=False)
            for i, segs in zip(keep, segments, strict=True):
                out[i] = turns_to_tensors(parse_segment(s) for s in segs)
        return out


def prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != SR:
        raise ValueError(f"SAMPLE_RATE must be 16000 (the model's rate), got {sample_rate}; resample first")
    return np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
