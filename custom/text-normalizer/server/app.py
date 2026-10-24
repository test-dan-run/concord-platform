"""text-normalizer: openai-inspired `text.transform` utility (no model, no weights)."""

from __future__ import annotations

import os

from aisvc import create_openai_inspired_app

from .normalize import Options, normalize

NAME = os.environ.get("AISVC_NAME", "text-normalizer")


def build_app():
    def handler(texts: list[str], options: Options) -> list[str]:
        return [normalize(t, options) for t in texts]

    return create_openai_inspired_app(NAME, "text.transform", handler, options_model=Options)
