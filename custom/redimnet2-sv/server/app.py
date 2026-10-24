"""redimnet2-sv: OIP v2 service. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> EMBEDDINGS (FP32 [-1, 192], L2-normalised; cosine = dot product).

Same signature as wavlm-sv, so callers can switch speaker embedders by service name.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .embedder import ReDimNet2Embedder

NAME = os.environ.get("AISVC_NAME", "redimnet2-sv")
VERSION = "b6-vb2-vox2-cnc2-lm-c5bbe0b"
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "EMBEDDINGS", "datatype": "FP32", "shape": [-1, -1]},
    ],
}


def build_app(model: ReDimNet2Embedder | None = None):
    model = model or ReDimNet2Embedder(os.environ.get("WEIGHTS_DIR", "/models"))

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
        rows = [model.embed(audio[i, : int(n)], sr) for i, n in enumerate(lengths)]
        return {"EMBEDDINGS": np.stack(rows).astype(np.float32)}

    return create_oip_app(NAME, SIGNATURE, handler, version=VERSION, platform="pytorch", load=model.load)
