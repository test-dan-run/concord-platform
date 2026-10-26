"""pyannote speaker-diarization-3.1, built from local checkpoints with pyannote.audio 4.x.

The pipeline batches inside each file: the segmentation model runs on 10 s windows at a 1 s step, and the
embedding model on every (window, speaker) pair, PYANNOTE_SEGMENTATION_BATCH and PYANNOTE_EMBEDDING_BATCH at a
time (default: the 3.1 config's 32 each). GPU memory depends on those two numbers, not on the file's length, so they are the
service's batch settings. Files in one request run one after another. On out of memory both numbers are halved
for the rest of the process and the file runs again.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .turns import turns_to_tensors

SEGMENTATION = "segmentation-3.0/pytorch_model.bin"
EMBEDDING = "wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
SEGMENTATION_BATCH = "PYANNOTE_SEGMENTATION_BATCH"  # environment variables; unset: the config's values
EMBEDDING_BATCH = "PYANNOTE_EMBEDDING_BATCH"
log = logging.getLogger("aisvc")


def batch_size(var: str, default: int) -> int:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return int(default)
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{var}={raw!r} must be a whole number of at least 1")
    return int(raw)


@dataclass(frozen=True)
class DiarParams:
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    exclusive: bool = False  # True: no overlapping turns (easier to align with ASR words)

    @classmethod
    def from_parameters(cls, params: dict) -> DiarParams:
        unknown = sorted(set(params) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; supported: {sorted(cls.__dataclass_fields__)}")
        p = cls(**params)
        for k in ("num_speakers", "min_speakers", "max_speakers"):
            v = getattr(p, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):  # JSON true is not 1
                raise ValueError(f"{k} must be a positive integer")
        if not isinstance(p.exclusive, bool):
            raise ValueError("exclusive must be true or false")
        return p


class PyannoteDiarizer:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.pipeline = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        import yaml
        from aisvc.gpu import limit_torch_memory
        from pyannote.audio import Model
        from pyannote.audio.core.plda import PLDA
        from pyannote.audio.pipelines import SpeakerDiarization

        for f in ("config.yaml", SEGMENTATION, EMBEDDING):
            if not (self.dir / f).is_file():
                raise FileNotFoundError(f"{self.dir / f} missing; run `just fetch-weights pyannote-diarization`")
        cfg = yaml.safe_load((self.dir / "config.yaml").read_text())
        p = cfg["pipeline"]["params"]
        # The 3.1 config names HF repos; load the same checkpoints from disk instead (no Hub access).
        # pyannote.audio 4 always resolves a PLDA (default: the gated community-1 repo) even though only
        # VBxClustering uses it; 3.1 uses AgglomerativeClustering, so an inert placeholder is passed.
        pipeline = SpeakerDiarization(
            segmentation=Model.from_pretrained(str(self.dir / SEGMENTATION)),
            embedding=Model.from_pretrained(str(self.dir / EMBEDDING)),
            clustering=p["clustering"],
            embedding_batch_size=batch_size(EMBEDDING_BATCH, p["embedding_batch_size"]),
            segmentation_batch_size=batch_size(SEGMENTATION_BATCH, p["segmentation_batch_size"]),
            embedding_exclude_overlap=p["embedding_exclude_overlap"],
            plda=PLDA.__new__(PLDA),
        )
        pipeline.instantiate(cfg["params"])
        if torch.cuda.is_available():
            limit_torch_memory(torch)
            pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

    def diarize(self, audio: np.ndarray, sample_rate: int, params: DiarParams) -> dict[str, np.ndarray]:
        import torch
        from aisvc.batch import free_cuda_cache, is_oom

        assert self.pipeline is not None, "call load() first"
        if not 8000 <= sample_rate <= 48000:
            raise ValueError(f"SAMPLE_RATE must be between 8000 and 48000, got {sample_rate}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < sample_rate // 10:  # under 0.1 s: nothing to diarize
            return turns_to_tensors([])
        # In-memory waveform: pyannote resamples to 16 kHz itself and never needs torchcodec/FFmpeg.
        file = {"waveform": torch.from_numpy(audio)[None], "sample_rate": sample_rate}
        pl = self.pipeline
        with self._lock:
            while True:
                try:
                    out = pl(file, num_speakers=params.num_speakers, min_speakers=params.min_speakers,
                             max_speakers=params.max_speakers)
                    break
                except Exception as exc:
                    if not is_oom(exc) or pl.segmentation_batch_size == pl.embedding_batch_size == 1:
                        raise
                free_cuda_cache()  # outside the except block, so the traceback's tensors are freed
                pl.segmentation_batch_size = max(1, pl.segmentation_batch_size // 2)
                pl.embedding_batch_size = max(1, pl.embedding_batch_size // 2)
                log.warning("out of memory; segmentation batch is now %d, embedding batch %d",
                            pl.segmentation_batch_size, pl.embedding_batch_size)
        ann = out.exclusive_speaker_diarization if params.exclusive else out.speaker_diarization
        return turns_to_tensors((t.start, t.end, label) for t, _, label in ann.itertracks(yield_label=True))
