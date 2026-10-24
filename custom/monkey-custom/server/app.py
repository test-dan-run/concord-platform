"""monkey-custom: openai-inspired `text.transform` transcript normaliser (transcript-normer @e98d4ae).

Removes verbal fillers and Singlish discourse particles, stitches spelled-out acronyms ("U S A" -> "USA"),
then formats by `mode`: `canonical` (whisper-normalizer: lowercase, number ITN, no punctuation),
`verbatim` (casing and number words kept) or `punctuated` (verbatim plus restored punctuation and casing
from an ONNX model). Self-corrections such as "five no wait ten" are never resolved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from aisvc import create_openai_inspired_app
from pydantic import BaseModel, ConfigDict

NAME = os.environ.get("AISVC_NAME", "monkey-custom")
MAX_INPUTS = 1000  # transcript-normer's MAX_BATCH_SIZE
MAX_CHARS = 100_000  # its MAX_TEXT_LENGTH
MODEL_FILES = ("punct_cap_seg_en.onnx", "spe_32k_lc_en.model")


class Options(BaseModel):
    """transcript-normer's request settings, as scalar text.transform options."""

    model_config = ConfigDict(extra="forbid")
    remove_fillers: bool = True
    remove_particles: bool = True
    mode: Literal["canonical", "verbatim", "punctuated"] = "canonical"


class Engine:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.normalizer = None

    def load(self) -> None:
        from .normer.normalizer import TranscriptNormalizer
        from .normer.punctuator import Punctuator

        for f in MODEL_FILES:
            if not (self.dir / f).is_file():
                raise FileNotFoundError(f"{self.dir / f} missing; run `just fetch-weights monkey-custom`")
        # download=False: weights are fetched and sha256-checked by `just fetch-weights`, never at start
        self.normalizer = TranscriptNormalizer(punctuator=Punctuator(self.dir, download=False))

    def __call__(self, texts: list[str], options: Options) -> list[str]:
        if len(texts) > MAX_INPUTS:
            raise ValueError(f"input has {len(texts)} items; at most {MAX_INPUTS}")
        if any(len(t) > MAX_CHARS for t in texts):
            raise ValueError(f"an input item is longer than {MAX_CHARS} characters")
        return self.normalizer.normalize_batch(texts, options.remove_fillers, options.remove_particles, options.mode)


def build_app(engine: Engine | None = None):
    engine = engine or Engine(os.environ.get("WEIGHTS_DIR", "/models"))
    return create_openai_inspired_app(NAME, "text.transform", engine, options_model=Options, load=engine.load)
