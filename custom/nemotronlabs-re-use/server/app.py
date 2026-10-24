"""nemotronlabs-re-use: OIP v2 service. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) -> AUDIO (FP32 [-1]) + SAMPLE_RATE.

Universal speech enhancement (noise, reverberation, clipping, band limits, codec artifacts, packet loss).
Output has the input's rate and length. Rates: 8, 16, 22.05, 24, 32, 44.1 and 48 kHz. No parameters.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .enhancer import ReUse

NAME = os.environ.get("AISVC_NAME", "nemotronlabs-re-use")
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


def build_app(model: ReUse | None = None):
    model = model or ReUse(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; this model takes none")
        sr = int(inputs["SAMPLE_RATE"].reshape(-1)[0])
        out = model.enhance(inputs["AUDIO"], sr)
        return {"AUDIO": out.astype(np.float32), "SAMPLE_RATE": np.array([sr], dtype=np.int32)}

    return create_oip_app(NAME, SIGNATURE, handler, version="1", platform="pytorch", load=model.load)
