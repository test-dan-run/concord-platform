"""faster-whisper-large-v3: OpenAI POST /v1/audio/transcriptions with word timestamps.

response_format json | text | verbose_json; with verbose_json, timestamp_granularities[]=word adds
`words` (segments are always included). Silero VAD filtering (bundled with faster-whisper) is on.
"""

from __future__ import annotations

import os

from aisvc import create_openai_app

from .model import FasterWhisper

NAME = os.environ.get("AISVC_NAME", "faster-whisper-large-v3")


def build_app(model: FasterWhisper | None = None):
    model = model or FasterWhisper(os.environ.get("WEIGHTS_DIR", "/models"))

    def handler(req):
        want_words = "word" in req.timestamp_granularities
        out = model.transcribe(req.file, language=req.language, prompt=req.prompt, temperature=req.temperature,
                               word_timestamps=want_words)
        if not want_words:
            out.pop("words", None)
        return out

    return create_openai_app(NAME, "audio.transcriptions", handler, load=model.load)
