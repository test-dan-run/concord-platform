"""wavlm-sv: OIP v2 service. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> EMBEDDINGS (FP32 [-1, 256], L2-normalised; cosine = dot product).

Same signature as redimnet2-sv, so callers can switch speaker embedders by service name.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import MicroBatcher, create_oip_app
from aisvc.batch import audio_ms

from .embedder import SR, WavLMEmbedder, prepare

NAME = os.environ.get("AISVC_NAME", "wavlm-sv")
VERSION = "large-ecapa-13ad4fc"
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


def build_app(model: WavLMEmbedder | None = None):
    model = model or WavLMEmbedder(os.environ.get("WEIGHTS_DIR", "/models"))
    # Any number of rows: sorted by length, run in batches within AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms of audio)
    batch = MicroBatcher(model.embed_batch, length=lambda x: audio_ms(x, SR), name="embed")

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
        rows = batch([prepare(audio[i, : int(n)], sr) for i, n in enumerate(lengths)])
        return {"EMBEDDINGS": np.stack(rows).astype(np.float32)}

    return create_oip_app(NAME, SIGNATURE, handler, version=VERSION, platform="pytorch", load=model.load)
