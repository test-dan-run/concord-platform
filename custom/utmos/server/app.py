"""utmos: OIP v2 service. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> MOS (FP32 [-1], UTMOS22 strong, about 1-5).

The batch signature matches dnsmos (minus its SIG/BAK outputs), so callers can switch between them.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .utmos import Utmos

NAME = os.environ.get("AISVC_NAME", "utmos")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "MOS", "datatype": "FP32", "shape": [-1]},
    ],
}


def build_app(model: Utmos | None = None):
    model = model or Utmos(os.environ.get("WEIGHTS_DIR", "/models"))

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
        scores = [model.score(audio[i, : int(n)], sr) for i, n in enumerate(lengths)]
        return {"MOS": np.asarray(scores, dtype=np.float32).reshape(-1)}

    return create_oip_app(NAME, SIGNATURE, handler, version="utmos22-strong-eecd3e6", platform="pytorch",
                          load=model.load)
