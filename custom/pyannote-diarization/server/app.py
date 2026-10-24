"""pyannote-diarization: OIP v2. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) -> SEGMENTS (FP32 [-1, 2]) + SPEAKERS (INT32 [-1]).

Optional request `parameters`: num_speakers, min_speakers, max_speakers, exclusive (no overlaps).
Speaker indices are in order of first appearance.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .diarizer import DiarParams, PyannoteDiarizer

NAME = os.environ.get("AISVC_NAME", "pyannote-diarization")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]},
        {"name": "SPEAKERS", "datatype": "INT32", "shape": [-1]},
    ],
}


def build_app(diarizer: PyannoteDiarizer | None = None):
    diarizer = diarizer or PyannoteDiarizer(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = DiarParams.from_parameters(parameters)
        return diarizer.diarize(inputs["AUDIO"], int(inputs["SAMPLE_RATE"].reshape(-1)[0]), params)

    return create_oip_app(NAME, SIGNATURE, handler, version="3.1", platform="pytorch", load=diarizer.load)
