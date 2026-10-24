"""wpe-dereverb: OIP v2 utility. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) -> AUDIO (FP32 [-1]) + SAMPLE_RATE.

Output has the input's rate and length. Optional request `parameters`: taps (10), delay (3), iterations (3).
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .dereverb import WpeParams, dereverb

NAME = os.environ.get("AISVC_NAME", "wpe-dereverb")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
}


def build_app():
    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = WpeParams.from_parameters(parameters)
        sr = int(inputs["SAMPLE_RATE"].reshape(-1)[0])
        return {"AUDIO": dereverb(inputs["AUDIO"], sr, params), "SAMPLE_RATE": np.array([sr], dtype=np.int32)}

    return create_oip_app(NAME, SIGNATURE, handler, version="0.0.11", platform="nara_wpe")
