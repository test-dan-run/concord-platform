"""voxcpm2: OIP v2 voice cloning. TEXT (BYTES [1]) + REF_AUDIO (FP32 [-1]) + REF_SAMPLE_RATE (INT32 [1]) +
REF_TEXT (BYTES [1]) + SEED (INT64 [1]) -> AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1], 48000).

Optional request `parameters`:
  mode                 "ultimate" (default: the clip as both continuation prompt and isolated reference),
                       "continuation" (prompt + transcript only) or "reference" (isolated reference; REF_TEXT unused)
  cfg_value            guidance scale, default 2.0
  inference_timesteps  diffusion steps, default 10
The same inputs and SEED give the same audio.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .synth import SynthParams, VoxCPM2Synth

NAME = os.environ.get("AISVC_NAME", "voxcpm2")
SIGNATURE = {
    "inputs": [
        {"name": "TEXT", "datatype": "BYTES", "shape": [1]},
        {"name": "REF_AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "REF_SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
        {"name": "REF_TEXT", "datatype": "BYTES", "shape": [1]},
        {"name": "SEED", "datatype": "INT64", "shape": [1]},
    ],
    "outputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
    ],
}


def _text(arr: np.ndarray) -> str:
    v = arr.reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def build_app(synth: VoxCPM2Synth | None = None):
    synth = synth or VoxCPM2Synth(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        params = SynthParams.from_parameters(parameters)
        audio, sr = synth.synthesize(_text(inputs["TEXT"]), inputs["REF_AUDIO"].reshape(-1),
                                     int(inputs["REF_SAMPLE_RATE"].reshape(-1)[0]), _text(inputs["REF_TEXT"]),
                                     int(inputs["SEED"].reshape(-1)[0]), params)
        return {"AUDIO": audio.astype(np.float32), "SAMPLE_RATE": np.array([sr], np.int32)}

    return create_oip_app(NAME, SIGNATURE, handler, version="2.0.3-32279ef", platform="pytorch", load=synth.load)
