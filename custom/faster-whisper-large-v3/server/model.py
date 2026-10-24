"""faster-whisper (CTranslate2) loaded from local files; returns OpenAI verbose_json-shaped dicts.

A file runs through faster-whisper's BatchedInferencePipeline: Silero VAD cuts the speech into windows of up to 30 s,
and the windows are decoded in batches of `batch_size()`. A window does not see the text of the window before it (no
conditioning on previous text, and no temperature fallback), so a long file runs several times faster with almost the
same words. Requests run one at a time: one batch of windows already fills the GPU, and memory stays predictable.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
from aisvc import Limits
from aisvc import batch as aisvc_batch

SAMPLE_RATE = 16000
WINDOW_MS = 30000  # Whisper pads every window to 30 s, so a batch of n windows is n x 30 s of padded audio


class FasterWhisper:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.model = None
        self.device = "cpu"
        self._lock = threading.Lock()
        # AISVC_MAX_BATCH windows per batch and at most AISVC_MAX_BATCH_LEN ms of padded windows; out of memory lowers
        # it for the rest of the process. Registered so /platform/memory and /platform/batch-limits see it.
        self.limits = Limits.from_env()
        aisvc_batch._LIMITS["transcribe"] = self.limits

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

    def batch_size(self) -> int:
        """Windows per batch: AISVC_MAX_BATCH, and no more 30 s windows than AISVC_MAX_BATCH_LEN allows."""
        lim = self.limits
        return max(1, min(lim.max_items, lim.max_len // WINDOW_MS if lim.max_len is not None else lim.max_items))

    def run_batched(self, audio: np.ndarray, **kwargs):
        """(segments, info) of one file. Out of memory halves the batch and tries again; a batch of one window that
        still runs out of memory raises. The caller holds the lock."""
        from faster_whisper import BatchedInferencePipeline

        while True:
            n = self.batch_size()
            try:
                # a new pipeline per call: it keeps the last word time of the file it works on
                segments, info = BatchedInferencePipeline(self.model).transcribe(audio, batch_size=n, **kwargs)
                return list(segments), info  # the generator runs the decoding
            except Exception as exc:
                if n == 1 or not aisvc_batch.is_oom(exc):
                    raise
            self.limits.lower(n * WINDOW_MS, n)  # max_len drops to n/2 windows

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None, prompt: str | None = None,
                   temperature: float | None = None, word_timestamps: bool = False, vad_filter: bool = True) -> dict:
        from faster_whisper.audio import decode_audio

        assert self.model is not None, "call load() first"
        try:
            audio = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # PyAV raises its own error types for undecodable input
            raise ValueError(f"could not decode audio file: {exc}") from None
        # without_timestamps=False: Whisper's timestamp tokens split each window into sentence-sized segments, as the
        # sequential decoder does (the pipeline's default is one segment per window)
        kwargs = {"language": language, "initial_prompt": prompt, "word_timestamps": word_timestamps,
                  "vad_filter": vad_filter, "beam_size": 5, "without_timestamps": False}
        if temperature is not None:
            kwargs["temperature"] = temperature
        with self._lock:
            segments, info = self.run_batched(audio, **kwargs)
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
