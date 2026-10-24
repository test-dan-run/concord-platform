"""ark-asr-0-6b: OpenAI POST /v1/audio/transcriptions (json | text | verbose_json).

The model detects the language itself; `language`, `prompt` and `temperature` are accepted and ignored (decoding is
fixed greedy, as the authors' script runs it). verbose_json adds `language: null` and `duration`, no segments.
"""

from __future__ import annotations

import os

from aisvc import create_openai_app

from .ark import ArkAsr

NAME = os.environ.get("AISVC_NAME", "ark-asr-0-6b")


def build_app(model: ArkAsr | None = None):
    model = model or ArkAsr(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(req):
        text, duration = model.transcribe(req.file)
        return {"task": "transcribe", "language": None, "duration": duration, "text": text}

    return create_openai_app(NAME, "audio.transcriptions", handler, load=model.load)
