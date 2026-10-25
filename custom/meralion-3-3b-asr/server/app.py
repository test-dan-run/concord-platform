"""meralion-3-3b-asr: OpenAI POST /v1/audio/transcriptions (json | text | verbose_json).

The model detects the language itself; `language`, `prompt` and `temperature` are accepted and ignored (decoding
is fixed greedy, as the MERaLiON team pins it). verbose_json adds `language: null` and `duration`, no segments.

Requests that arrive together are transcribed together: each file is cut into 30 s chunks in its own request thread,
and the chunks of every waiting request run through one batched `generate` per AISVC_MAX_BATCH chunks.
"""

from __future__ import annotations

import os
from itertools import islice

import numpy as np
from aisvc import DynamicBatcher, Limits, MicroBatcher, create_openai_app

from .meralion import CHUNK_MS, SR, Meralion3Asr, chunks, decode_audio, join

NAME = os.environ.get("AISVC_NAME", "meralion-3-3b-asr")


def build_app(model: Meralion3Asr | None = None):
    model = model or Meralion3Asr(os.environ.get("WEIGHTS_DIR", "/models"))
    # Chunks in batches within AISVC_MAX_BATCH / AISVC_MAX_BATCH_LEN (ms; every chunk counts as a full 30 s window)
    pieces = MicroBatcher(model.transcribe_batch, length=lambda x: CHUNK_MS, name="transcribe")

    def files(group: list[list[np.ndarray]]) -> list[str]:
        texts = iter(pieces([c for f in group for c in f]))
        return [join(islice(texts, len(f))) for f in group]

    # Requests within AISVC_BATCH_WAIT_MS of each other share one call. The model takes no options, so every
    # request can share; the limits apply to the chunks above, not to the number of requests.
    transcribe = DynamicBatcher(files, length=len, limits=Limits(max_items=1 << 16), name="requests")

    def handler(req):
        x = decode_audio(req.file)  # here, so a bad file fails only its own request
        return {"task": "transcribe", "language": None, "duration": round(len(x) / SR, 3), "text": transcribe(chunks(x))}

    return create_openai_app(NAME, "audio.transcriptions", handler, load=model.load)
