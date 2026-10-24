"""silero-vad: OIP v2 service. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) -> SEGMENTS (FP32 [-1, 2]).

Optional request `parameters`: threshold, neg_threshold, min_speech_duration_ms,
min_silence_duration_ms, speech_pad_ms (Silero defaults otherwise).
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .vad import SileroVAD, VadParams

NAME = os.environ.get("AISVC_NAME", "silero-vad")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
    "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]}],
}


def build_app(vad: SileroVAD | None = None):
    vad = vad or SileroVAD(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = VadParams.from_parameters(parameters)
        sample_rate = int(inputs["SAMPLE_RATE"].reshape(-1)[0])
        return {"SEGMENTS": vad.segments(inputs["AUDIO"], sample_rate, params)}

    return create_oip_app(NAME, SIGNATURE, handler, version="6.2.3", platform="onnxruntime", load=vad.load)
