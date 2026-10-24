"""nemotron-3-diarization: OIP v2. AUDIO (FP32 [-1], 16 kHz) + SAMPLE_RATE (INT32 [1]) -> SEGMENTS (FP32 [-1, 2]) + SPEAKERS (INT32 [-1]).

Same signature as pyannote-diarization; speaker indices are in order of first appearance (Sortformer's
arrival order), at most 8 speakers. No request parameters.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .diarizer import NemotronDiarizer

NAME = os.environ.get("AISVC_NAME", "nemotron-3-diarization")
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


def build_app(diarizer: NemotronDiarizer | None = None):
    diarizer = diarizer or NemotronDiarizer(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; this service takes none")
        return diarizer.diarize(inputs["AUDIO"], int(inputs["SAMPLE_RATE"].reshape(-1)[0]))

    return create_oip_app(NAME, SIGNATURE, handler, version="3", platform="nemo", load=diarizer.load)
