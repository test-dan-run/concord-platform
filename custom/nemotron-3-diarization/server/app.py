"""nemotron-3-diarization: OIP v2. AUDIO (FP32 [-1, -1], zero-padded batch, 16 kHz) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> SEGMENTS (FP32 [-1, 2]) + SPEAKERS (INT32 [-1]) + ROWS (INT32 [-1]).

Same signature as pyannote-diarization. The outputs hold the segments of every row, row after row, each row's
sorted by start; ROWS gives the row of each segment. Speaker indices are per row, in order of first appearance
(Sortformer's arrival order), at most 8 speakers. No request parameters. The single-clip request (AUDIO FP32 [-1],
no LENGTHS) still works and gets SEGMENTS and SPEAKERS without ROWS (server/rows.py).
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import MicroBatcher, create_oip_app
from aisvc.batch import audio_ms

from .diarizer import SR, NemotronDiarizer, prepare
from .rows import accept_single_clip, rows
from .turns import stack_rows

NAME = os.environ.get("AISVC_NAME", "nemotron-3-diarization")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]},
        {"name": "SPEAKERS", "datatype": "INT32", "shape": [-1]},
        {"name": "ROWS", "datatype": "INT32", "shape": [-1]},
    ],
}


def build_app(diarizer: NemotronDiarizer | None = None):
    diarizer = diarizer or NemotronDiarizer(os.environ.get("WEIGHTS_DIR", "/models"))
    # Any number of rows: sorted by length, run in padded batches within AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms of audio)
    batch = MicroBatcher(diarizer.diarize_batch, length=lambda x: audio_ms(x, SR), name="diarize")

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; this service takes none")
        clips, sr = rows(inputs)
        return stack_rows(batch([prepare(c, sr) for c in clips]))

    app = create_oip_app(NAME, SIGNATURE, handler, version="3", platform="nemo", load=diarizer.load)
    return accept_single_clip(app, drop={"ROWS"})
