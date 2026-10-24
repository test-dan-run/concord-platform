"""Batch requests for the enhancer and diarizer signatures, with the old single-clip request still accepted.

Identical in every enhancer and diarization service (nemotronlabs-re-use, deepfilternet3, wpe-dereverb,
pyannote-diarization, nemotron-3-diarization); tests/test_images.py checks that the copies match.

Batch form: AUDIO FP32 [N, T] zero-padded, LENGTHS INT32 [N], SAMPLE_RATE INT32 [1] (one rate for every row).
Single-clip form, which callers sent before the batch signatures: AUDIO FP32 [T] and no LENGTHS.
`accept_single_clip` takes it as one row and answers in the old shapes: outputs named in `squeeze` lose their
row axis ([1, T] -> [T]) and outputs named in `drop` are left out.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import numpy as np
from aisvc.oip_app import InferenceRequest, RequestInput
from fastapi import FastAPI
from fastapi.routing import APIRoute


def rows(inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], int]:
    """(each AUDIO row cut to its LENGTHS entry, as float32, and the sample rate) of a batch request."""
    audio, lengths = inputs["AUDIO"], inputs["LENGTHS"].reshape(-1)
    if audio.shape[0] != len(lengths):
        raise ValueError(f"AUDIO has {audio.shape[0]} rows but LENGTHS has {len(lengths)} entries")
    if np.any(lengths < 0) or np.any(lengths > audio.shape[1]):
        raise ValueError("every LENGTHS entry must be in [0, AUDIO.shape[1]]")
    clips = [np.ascontiguousarray(audio[i, : int(n)], dtype=np.float32) for i, n in enumerate(lengths)]
    return clips, int(inputs["SAMPLE_RATE"].reshape(-1)[0])


def accept_single_clip(app: FastAPI, *, squeeze: Collection[str] = (), drop: Collection[str] = ()) -> FastAPI:
    """Let the app's infer routes also take the single-clip form (see the module docstring)."""
    routes = app.router.routes
    for i, route in enumerate(routes):
        if isinstance(route, APIRoute) and route.path.endswith("/infer"):
            app.router.add_api_route(route.path, _single_clip(route.endpoint, set(squeeze), set(drop)), methods=["POST"])
            routes[i] = routes.pop()  # in place of the batch-only route
    return app


def _single_clip(infer: Callable[..., dict], squeeze: set[str], drop: set[str]) -> Callable[..., dict]:
    def endpoint(model: str, body: InferenceRequest, ver: str | None = None) -> dict:
        audio = next((t for t in body.inputs if t.name == "AUDIO"), None)
        if audio is None or len(audio.shape) != 1 or any(t.name == "LENGTHS" for t in body.inputs):
            return infer(model, body, ver)  # the batch form, or a request the signature check will reject
        n = audio.shape[0]
        inputs = [t.model_copy(update={"shape": [1, n]}) if t is audio else t for t in body.inputs]
        lengths = RequestInput(name="LENGTHS", datatype="INT32", shape=[1], data=[n])
        out = infer(model, body.model_copy(update={"inputs": [*inputs, lengths]}), ver)
        out["outputs"] = [{**o, "shape": o["shape"][1:]} if o["name"] in squeeze else o
                          for o in out["outputs"] if o["name"] not in drop]
        return out

    return endpoint
