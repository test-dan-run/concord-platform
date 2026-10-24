"""kokoro-tts: OpenAI POST /v1/audio/speech backed by Kokoro-82M (24 kHz mono)."""

from __future__ import annotations

import os

from aisvc import create_openai_app

from .model import KokoroModel

NAME = os.environ.get("AISVC_NAME", "kokoro-tts")


def build_app(model: KokoroModel | None = None):
    model = model or KokoroModel(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(req):
        return model.synthesize(req.input, voice=req.voice, speed=req.speed)

    return create_openai_app(NAME, "audio.speech", handler, load=model.load)
