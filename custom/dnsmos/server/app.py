"""dnsmos: OIP v2 service. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> MOS (FP32 [-1], DNSMOS OVRL), SIG (FP32 [-1]), BAK (FP32 [-1]).

The batch signature matches the other clip scorers (utmos), so callers can switch between them.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import Limits, MicroBatcher, create_oip_app
from aisvc.batch import audio_ms

from .dnsmos import SR, Dnsmos, prepare

NAME = os.environ.get("AISVC_NAME", "dnsmos")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "MOS", "datatype": "FP32", "shape": [-1]},
        {"name": "SIG", "datatype": "FP32", "shape": [-1]},
        {"name": "BAK", "datatype": "FP32", "shape": [-1]},
    ],
}


def build_app(model: Dnsmos | None = None):
    model = model or Dnsmos(os.environ.get("WEIGHTS_DIR", "/models"))
    # Any number of rows, run in batches within AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms of audio). On the CPU a
    # batch is no faster (6.6 s one by one against 13.7 s in one batch for 32 clips of 2 to 15 s, measured 2026-10-24), so the default is one clip at a time.
    batch = MicroBatcher(model.score_batch, length=lambda x: audio_ms(x, SR), limits=Limits.from_env(max_items=1), name="score")

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; none are supported")
        audio, lengths = inputs["AUDIO"], inputs["LENGTHS"].reshape(-1)
        sr = int(inputs["SAMPLE_RATE"].reshape(-1)[0])
        if sr < 8000:
            raise ValueError(f"SAMPLE_RATE must be at least 8000, got {sr}")
        if audio.shape[0] != len(lengths):
            raise ValueError(f"AUDIO has {audio.shape[0]} rows but LENGTHS has {len(lengths)} entries")
        if np.any(lengths <= 0) or np.any(lengths > audio.shape[1]):
            raise ValueError("every LENGTHS entry must be in [1, AUDIO.shape[1]]")
        scores = batch([prepare(audio[i, : int(n)], sr) for i, n in enumerate(lengths)])
        arr = np.asarray(scores, dtype=np.float32).reshape(-1, 3)
        return {"MOS": arr[:, 0], "SIG": arr[:, 1], "BAK": arr[:, 2]}

    return create_oip_app(NAME, SIGNATURE, handler, version="p835-82f1b17", platform="onnxruntime", load=model.load)
