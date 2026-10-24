"""Kokoro-82M loaded from local files only (no Hugging Face calls at runtime)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

REPO_ID = "hexgrad/Kokoro-82M"  # only used by kokoro to pick v1.0 behaviour; files come from disk
SAMPLE_RATE = 24000
MODEL_FILE = "kokoro-v1_0.pth"
LANGS = ("a", "b")  # American and British English pipelines; voice ids start with the lang code

# OpenAI voice names -> Kokoro voices, so OpenAI SDK defaults work. Kokoro ids also work directly.
OPENAI_VOICES = {
    "alloy": "af_alloy", "ash": "am_adam", "ballad": "bm_george", "coral": "af_heart",
    "echo": "am_echo", "fable": "bm_fable", "nova": "af_nova", "onyx": "am_onyx",
    "sage": "af_sarah", "shimmer": "af_bella", "verse": "am_michael",
}


class KokoroModel:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.pipelines: dict = {}
        self.device = "cpu"
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from aisvc.gpu import limit_torch_memory
        from kokoro import KModel, KPipeline

        for f in ("config.json", MODEL_FILE):
            if not (self.dir / f).is_file():
                raise FileNotFoundError(f"{self.dir / f} missing; run `just fetch-weights kokoro-tts`")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        limit_torch_memory(torch)
        model = KModel(repo_id=REPO_ID, config=str(self.dir / "config.json"), model=str(self.dir / MODEL_FILE))
        model = model.to(self.device).eval()
        self.pipelines = {lang: KPipeline(lang_code=lang, repo_id=REPO_ID, model=model) for lang in LANGS}
        self.synthesize("Ready.", "af_heart")  # warm-up: G2P, CUDA kernels

    def voices(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "voices").glob("*.pt"))

    def resolve_voice(self, voice: str) -> tuple[str, Path]:
        vid = OPENAI_VOICES.get(voice, voice)
        path = self.dir / "voices" / f"{vid}.pt"
        if not path.is_file() or vid[:1] not in self.pipelines:
            usable = [v for v in self.voices() if v[:1] in self.pipelines]
            raise ValueError(f"unknown voice {voice!r}; use an OpenAI voice ({', '.join(sorted(OPENAI_VOICES))}) "
                             f"or a Kokoro voice ({', '.join(usable)})")
        return vid[:1], path

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import torch

        lang, path = self.resolve_voice(voice)
        with self._lock, torch.inference_mode():
            chunks = [r.audio.detach().cpu().numpy() for r in self.pipelines[lang](text, voice=str(path), speed=speed)
                      if r.audio is not None]
        if not chunks:
            raise ValueError("no audio produced; the input has no pronounceable text")
        return np.concatenate(chunks).astype(np.float32), SAMPLE_RATE
