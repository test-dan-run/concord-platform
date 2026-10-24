"""qwen3-forced-aligner: OIP v2 forced alignment. AUDIO (FP32 [-1]) + SAMPLE_RATE (INT32 [1]) + TEXT (BYTES [1]) +
LANGUAGE (BYTES [1], ISO code or name, e.g. "en" / "English") -> WORDS (BYTES [-1]), TIMES (FP32 [-1, 2], start/end
seconds), SCORES (FP32 [-1], 0-1 confidence of each word's two timestamps).

WORDS are the transcript's words as the aligner tokenises them (punctuation dropped, apostrophes kept; Chinese
characters one by one). Send at most 180 s of audio per request.
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import create_oip_app

from .aligner import QwenForcedAligner

NAME = os.environ.get("AISVC_NAME", "qwen3-forced-aligner")
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
        {"name": "TEXT", "datatype": "BYTES", "shape": [1]},
        {"name": "LANGUAGE", "datatype": "BYTES", "shape": [1]},
    ],
    "outputs": [
        {"name": "WORDS", "datatype": "BYTES", "shape": [-1]},
        {"name": "TIMES", "datatype": "FP32", "shape": [-1, 2]},
        {"name": "SCORES", "datatype": "FP32", "shape": [-1]},
    ],
}


def _text(arr: np.ndarray) -> str:
    v = arr.reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def build_app(aligner: QwenForcedAligner | None = None):
    aligner = aligner or QwenForcedAligner(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; none are supported")
        words, times, scores = aligner.align(inputs["AUDIO"].reshape(-1), int(inputs["SAMPLE_RATE"].reshape(-1)[0]),
                                             _text(inputs["TEXT"]), _text(inputs["LANGUAGE"]))
        return {"WORDS": np.array(words, dtype=object), "TIMES": np.asarray(times, np.float32).reshape(-1, 2),
                "SCORES": np.asarray(scores, np.float32).reshape(-1)}

    return create_oip_app(NAME, SIGNATURE, handler, version="0.6b-c7cbfc2", platform="pytorch", load=aligner.load)
