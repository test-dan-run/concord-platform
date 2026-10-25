"""qwen3-forced-aligner: OIP v2 forced alignment of a batch of clips.

    AUDIO FP32 [-1, -1] + LENGTHS INT32 [-1]   N clips, padded; LENGTHS gives each row's samples
    SAMPLE_RATE INT32 [1]                      one rate for every row
    TEXT BYTES [-1], LANGUAGE BYTES [-1]       one entry per row, or one entry for every row (ISO code or name)
 -> WORDS BYTES [-1], TIMES FP32 [-1, 2] (start/end seconds), SCORES FP32 [-1] (0-1 confidence of each word's two
    timestamps), ROWS INT32 [-1] (the input row each word belongs to; the words of row 0 come first, then row 1, ...)

WORDS are the transcript's words as the aligner tokenises them (punctuation dropped, apostrophes kept; Chinese
characters one by one). Each row is at most 180 s. Rows are sorted by length and aligned in padded batches within
AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms of audio).

The single-clip request of 0.1.0 still works: AUDIO [-1] without LENGTHS, with TEXT and LANGUAGE [1], runs as one
row and gets the 0.1.0 outputs (WORDS, TIMES, SCORES; no ROWS unless asked for).
"""

from __future__ import annotations

import os

import numpy as np
from aisvc import MicroBatcher, create_oip_app
from aisvc.batch import audio_ms
from aisvc.oip_app import InferenceRequest, RequestInput, RequestOutput

from .aligner import SR, QwenForcedAligner

NAME = os.environ.get("AISVC_NAME", "qwen3-forced-aligner")
VERSION = "0.6b-c7cbfc2"
SIGNATURE = {
    "inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]},
        {"name": "TEXT", "datatype": "BYTES", "shape": [-1]},
        {"name": "LANGUAGE", "datatype": "BYTES", "shape": [-1]},
    ],
    "outputs": [
        {"name": "WORDS", "datatype": "BYTES", "shape": [-1]},
        {"name": "TIMES", "datatype": "FP32", "shape": [-1, 2]},
        {"name": "SCORES", "datatype": "FP32", "shape": [-1]},
        {"name": "ROWS", "datatype": "INT32", "shape": [-1]},
    ],
}
SINGLE_OUTPUTS = ("WORDS", "TIMES", "SCORES")  # what a 0.1.0 single-clip request gets back
INFER_PATHS = ("/v2/models/{model}/infer", "/v2/models/{model}/versions/{ver}/infer")


def _texts(arr: np.ndarray, n: int, name: str) -> list[str]:
    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in arr.reshape(-1)]
    if len(vals) not in (1, n):
        raise ValueError(f"{name} has {len(vals)} entries for {n} rows; give one per row, or one for every row")
    return vals * n if len(vals) == 1 else vals


def accept_single_clip(app) -> None:
    """Keep the 0.1.0 request working: AUDIO [-1] without LENGTHS becomes one row of [1, -1] before the signature
    check, and the response has the 0.1.0 outputs unless the request names its outputs."""
    infer = next(r.endpoint for r in app.router.routes if getattr(r, "path", None) == INFER_PATHS[0])
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in INFER_PATHS]

    def infer_single_or_batch(model: str, body: InferenceRequest, ver: str | None = None) -> dict:
        given = {i.name: i for i in body.inputs}
        audio = given.get("AUDIO")
        if audio is not None and len(audio.shape) == 1 and "LENGTHS" not in given:
            audio.shape = [1, audio.shape[0]]
            body.inputs.append(RequestInput(name="LENGTHS", datatype="INT32", shape=[1], data=[audio.shape[1]]))
            body.outputs = body.outputs or [RequestOutput(name=n) for n in SINGLE_OUTPUTS]
        return infer(model, body, ver)

    for path in INFER_PATHS:
        app.post(path)(infer_single_or_batch)


def build_app(aligner: QwenForcedAligner | None = None):
    aligner = aligner or QwenForcedAligner(os.environ.get("WEIGHTS_DIR", "/models"))
    # Rows sorted by length, in padded batches within AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms of audio)
    batch = MicroBatcher(aligner.slot_classes_batch, length=lambda r: audio_ms(r.audio, SR), name="align")

    def handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]:
        if parameters:
            raise ValueError(f"unknown parameters {sorted(parameters)}; none are supported")
        audio, lengths = inputs["AUDIO"], inputs["LENGTHS"].reshape(-1)
        sr, n = int(inputs["SAMPLE_RATE"].reshape(-1)[0]), audio.shape[0]
        if n == 0 or len(lengths) != n:
            raise ValueError(f"AUDIO has {n} rows but LENGTHS has {len(lengths)} entries" if n else "AUDIO has no rows")
        if np.any(lengths <= 0) or np.any(lengths > audio.shape[1]):
            raise ValueError("every LENGTHS entry must be in [1, AUDIO.shape[1]]")
        texts, langs = _texts(inputs["TEXT"], n, "TEXT"), _texts(inputs["LANGUAGE"], n, "LANGUAGE")
        rows = [aligner.row(audio[i, : int(k)], sr, texts[i], langs[i]) for i, k in enumerate(lengths)]
        results = aligner.align_rows(rows, batch)
        return {"WORDS": np.array([w for r in results for w in r[0]], dtype=object),
                "TIMES": np.concatenate([r[1] for r in results]).astype(np.float32).reshape(-1, 2),
                "SCORES": np.concatenate([r[2] for r in results]).astype(np.float32),
                "ROWS": np.concatenate([np.full(len(r[0]), i, np.int32) for i, r in enumerate(results)])}

    app = create_oip_app(NAME, SIGNATURE, handler, version=VERSION, platform="pytorch", load=aligner.load)
    accept_single_clip(app)
    return app
