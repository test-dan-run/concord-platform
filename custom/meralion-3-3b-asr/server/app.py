"""meralion-3-3b-asr: OpenAI POST /v1/audio/transcriptions (json | text | verbose_json).

The model detects the language itself; `language`, `prompt` and `temperature` are accepted and ignored (decoding
is fixed greedy, as the MERaLiON team pins it). verbose_json adds `language: null` and `duration`, no segments.
"""

from __future__ import annotations

import os

from aisvc import create_openai_app

from .meralion import Meralion3Asr

NAME = os.environ.get("AISVC_NAME", "meralion-3-3b-asr")


def build_app(model: Meralion3Asr | None = None):
    model = model or Meralion3Asr(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(req):
        text, duration = model.transcribe(req.file)
        return {"task": "transcribe", "language": None, "duration": duration, "text": text}

    return create_openai_app(NAME, "audio.transcriptions", handler, load=model.load)
