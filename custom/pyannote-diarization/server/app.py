"""pyannote-diarization: OIP v2. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE (INT32 [1])
-> SEGMENTS (FP32 [-1, 2]) + SPEAKERS (INT32 [-1]) + ROWS (INT32 [-1]).

The outputs hold the segments of every row, row after row, each row's sorted by start; ROWS gives the row of each
segment. Speaker indices are per row, in order of first appearance. Optional request `parameters`, for every row:
num_speakers, min_speakers, max_speakers, exclusive (no overlaps). Rows run one after another; the pipeline batches
inside each file (server/diarizer.py). The single-clip request (AUDIO FP32 [-1], no LENGTHS) still works and gets
SEGMENTS and SPEAKERS without ROWS (server/rows.py). Same signature as nemotron-3-diarization.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .diarizer import DiarParams, PyannoteDiarizer
from .rows import accept_single_clip, rows
from .turns import stack_rows

NAME = os.environ.get("AISVC_NAME", "pyannote-diarization")
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


def build_app(diarizer: PyannoteDiarizer | None = None):
    diarizer = diarizer or PyannoteDiarizer(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = DiarParams.from_parameters(parameters)
        clips, sr = rows(inputs)
        return stack_rows([diarizer.diarize(c, sr, params) for c in clips])

    app = create_oip_app(NAME, SIGNATURE, handler, version="3.1", platform="pytorch", load=diarizer.load)
    return accept_single_clip(app, drop={"ROWS"})
