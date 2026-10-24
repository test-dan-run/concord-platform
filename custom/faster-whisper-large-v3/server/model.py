"""faster-whisper (CTranslate2) loaded from local files; returns OpenAI verbose_json-shaped dicts."""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class FasterWhisper:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.model = None
        self.device = "cpu"
        self._lock = threading.Lock()

    def load(self) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        if not (self.dir / "model.bin").is_file():
            raise FileNotFoundError(f"{self.dir / 'model.bin'} missing; run `just fetch-weights faster-whisper-large-v3`")
        forced = os.environ.get("ASR_DEVICE")
        if forced:
            self.model = self._open(WhisperModel, forced)
            return
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                self.model = self._open(WhisperModel, "cuda")
                # CTranslate2 loads cuBLAS on first use, so a broken CUDA setup only fails here, not above.
                list(self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])
                return
            except Exception as exc:  # no cuBLAS, driver too old, out of VRAM, ...
                self.model = None
                print(f"faster-whisper: CUDA unusable ({exc}); falling back to CPU", file=sys.stderr, flush=True)
        self.model = self._open(WhisperModel, "cpu")

    def _open(self, whisper_model, device: str):
        # float16 on GPU: CTranslate2 disables INT8 on sm_120 (Blackwell); int8 is the fast CPU path.
        compute = os.environ.get("ASR_COMPUTE_TYPE") or ("float16" if device == "cuda" else "int8")
        model = whisper_model(str(self.dir), device=device, compute_type=compute)
        self.device = device
        print(f"faster-whisper: loaded on {device} ({compute})", file=sys.stderr, flush=True)
        return model

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None, prompt: str | None = None,
                   temperature: float | None = None, word_timestamps: bool = False, vad_filter: bool = True) -> dict:
        from faster_whisper.audio import decode_audio

        assert self.model is not None, "call load() first"
        try:
            audio = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # PyAV raises its own error types for undecodable input
            raise ValueError(f"could not decode audio file: {exc}") from None
        kwargs = {"language": language, "initial_prompt": prompt, "word_timestamps": word_timestamps,
                  "vad_filter": vad_filter, "beam_size": 5}
        if temperature is not None:
            kwargs["temperature"] = temperature
        with self._lock:
            segments, info = self.model.transcribe(audio, **kwargs)
            segments = list(segments)  # the generator runs the decoding
        return {
            "task": "transcribe",
            "language": info.language,
            "duration": round(float(info.duration), 3),
            "text": "".join(s.text for s in segments).strip(),
            "segments": [{
                "id": s.id, "seek": s.seek, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                "tokens": list(s.tokens), "temperature": s.temperature, "avg_logprob": s.avg_logprob,
                "compression_ratio": s.compression_ratio, "no_speech_prob": s.no_speech_prob,
            } for s in segments],
            "words": [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "probability": round(w.probability, 4)}
                      for s in segments for w in (s.words or [])],
        }
