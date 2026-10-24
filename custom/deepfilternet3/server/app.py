"""deepfilternet3: OIP v2 service. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) -> AUDIO (FP32 [-1]) + SAMPLE_RATE.

Output has the input's rate and length. Optional request `parameters`: atten_lim_db (0-100, default 100),
post_filter (bool, default false).
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .deepfilter import DeepFilter, DfParams

NAME = os.environ.get("AISVC_NAME", "deepfilternet3")
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


def build_app(model: DeepFilter | None = None):
    model = model or DeepFilter(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = DfParams.from_parameters(parameters)
        sr = int(inputs["SAMPLE_RATE"].reshape(-1)[0])
        out = model.enhance(inputs["AUDIO"], sr, params)
        return {"AUDIO": out, "SAMPLE_RATE": np.array([sr], dtype=np.int32)}

    return create_oip_app(NAME, SIGNATURE, handler, version="0.5.6", platform="tract", load=model.load)
