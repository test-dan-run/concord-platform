"""wpe-dereverb: OIP v2 utility. AUDIO (FP32 [-1, -1], zero-padded batch) + LENGTHS (INT32 [-1]) + SAMPLE_RATE
(INT32 [1]) -> AUDIO (FP32 [-1, -1], zero-padded) + LENGTHS (INT32 [-1]) + SAMPLE_RATE.

Each output row has its input row's rate and length. Optional request `parameters`, for every row: taps (10),
delay (3), iterations (3). Rows run one after another. The single-clip request (AUDIO FP32 [-1], no LENGTHS) still
works and gets AUDIO FP32 [-1] back (server/rows.py). Same signature as nemotronlabs-re-use and deepfilternet3.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app
from aisvc.batch import pad

from .dereverb import WpeParams, dereverb
from .rows import accept_single_clip, rows

NAME = os.environ.get("AISVC_NAME", "wpe-dereverb")
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


def build_app():
    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = WpeParams.from_parameters(parameters)
        clips, sr = rows(inputs)
        audio, lengths = pad([dereverb(c, sr, params) for c in clips])
        return {"AUDIO": audio, "LENGTHS": lengths, "SAMPLE_RATE": np.array([sr], dtype=np.int32)}

    app = create_oip_app(NAME, SIGNATURE, handler, version="0.0.11", platform="nara_wpe")
    return accept_single_clip(app, squeeze={"AUDIO"}, drop={"LENGTHS"})
