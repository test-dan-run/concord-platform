"""nemotronlabs-re-use: OIP v2 service. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> AUDIO (FP32 [-1, -1], zero-padded) + LENGTHS (INT32 [-1]) + SAMPLE_RATE.

Universal speech enhancement (noise, reverberation, clipping, band limits, codec artifacts, packet loss).
Each output row has its input row's rate and length. Rates: 8, 16, 22.05, 24, 32, 44.1 and 48 kHz. No parameters.
The single-clip request (AUDIO FP32 [-1], no LENGTHS) still works and gets AUDIO FP32 [-1] back (server/rows.py).
Same signature as deepfilternet3 and wpe-dereverb.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import MicroBatcher, create_oip_app
from aisvc.batch import pad

from .enhancer import ReUse
from .rows import accept_single_clip, rows

NAME = os.environ.get("AISVC_NAME", "nemotronlabs-re-use")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
}


def build_app(model: ReUse | None = None):
    model = model or ReUse(os.environ.get("WEIGHTS_DIR", "/models"))
    # Pieces (whole inputs, or the 5 s chunks of long ones) sorted by length and run in batches within
    # AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN. The length unit is samples / 16 (ms at 16 kHz): memory grows with
    # samples at any rate, and the catalog measures the batch profile at 16 kHz.
    batch = MicroBatcher(model.enhance_batch, length=lambda p: len(p[0]) // 16, name="enhance")

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; this model takes none")
        clips, sr = rows(inputs)
        audio, lengths = pad(model.enhance_rows(clips, sr, run=batch))
        return {"AUDIO": audio, "LENGTHS": lengths, "SAMPLE_RATE": np.array([sr], dtype=np.int32)}

    app = create_oip_app(NAME, SIGNATURE, handler, version="1", platform="pytorch", load=model.load)
    return accept_single_clip(app, squeeze={"AUDIO"}, drop={"LENGTHS"})
