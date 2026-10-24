"""Silero VAD v6 (ONNX, CPU): per-frame speech probabilities and speech segments.

Mirrors silero_vad.utils_vad (v6.2.3) OnnxWrapper + get_speech_timestamps without torch:
frames of 512 samples at 16 kHz (256 at 8 kHz) are fed with 64 (32) samples of context and a
recurrent state of shape (2, 1, 128). Segment logic follows get_speech_timestamps_from_probs with
the default max_speech_duration_s=inf.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_FILE = "silero_vad.onnx"


@dataclass(frozen=True)
class VadParams:
    threshold: float = 0.5
    neg_threshold: float | None = None  # default max(threshold - 0.15, 0.01)
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30

    @classmethod
    def from_parameters(cls, params: dict) -> VadParams:
        known = {f: params[f] for f in cls.__dataclass_fields__ if f in params}
        unknown = sorted(set(params) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; supported: {sorted(cls.__dataclass_fields__)}")
        p = cls(**known)
        if not 0.0 < p.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        return p


def normalize_rate(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    """Silero supports 8 kHz and 16 kHz; multiples of 16 kHz are decimated. Returns (audio, sr, step)."""
    if sample_rate > 16000 and sample_rate % 16000 == 0:
        step = sample_rate // 16000
        return audio[::step], 16000, step
    if sample_rate not in (8000, 16000):
        raise ValueError(f"SAMPLE_RATE must be 8000, 16000 or a multiple of 16000, got {sample_rate}")
    return audio, sample_rate, 1


def speech_segments(probs: np.ndarray, sample_rate: int, n_samples: int, p: VadParams) -> list[tuple[int, int]]:
    """Speech (start, end) in samples from per-frame probabilities."""
    window = 512 if sample_rate == 16000 else 256
    min_speech = sample_rate * p.min_speech_duration_ms / 1000
    pad = sample_rate * p.speech_pad_ms / 1000
    min_silence = sample_rate * p.min_silence_duration_ms / 1000
    neg = p.neg_threshold if p.neg_threshold is not None else max(p.threshold - 0.15, 0.01)

    triggered, temp_end, start = False, 0, 0
    speeches: list[list[int]] = []
    for i, prob in enumerate(probs):
        cur = window * i
        if prob >= p.threshold and temp_end:
            temp_end = 0
        if prob >= p.threshold and not triggered:
            triggered, start = True, cur
            continue
        if prob < neg and triggered:
            if not temp_end:
                temp_end = cur
            if cur - temp_end < min_silence:
                continue
            if temp_end - start > min_speech:
                speeches.append([start, temp_end])
            triggered, temp_end = False, 0
    if triggered and n_samples - start > min_speech:
        speeches.append([start, n_samples])

    for i, sp in enumerate(speeches):
        if i == 0:
            sp[0] = int(max(0, sp[0] - pad))
        if i != len(speeches) - 1:
            gap = speeches[i + 1][0] - sp[1]
            if gap < 2 * pad:
                sp[1] += int(gap // 2)
                speeches[i + 1][0] = int(max(0, speeches[i + 1][0] - gap // 2))
            else:
                sp[1] = int(min(n_samples, sp[1] + pad))
                speeches[i + 1][0] = int(max(0, speeches[i + 1][0] - pad))
        else:
            sp[1] = int(min(n_samples, sp[1] + pad))
    return [(a, b) for a, b in speeches]


class SileroVAD:
    def __init__(self, weights_dir: str | Path):
        self.path = Path(weights_dir) / MODEL_FILE
        self.session = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import onnxruntime as ort

        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} missing; run `just fetch-weights silero-vad`")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=["CPUExecutionProvider"])

    def speech_probs(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self.session is not None, "call load() first"
        window = 512 if sample_rate == 16000 else 256
        context_size = 64 if sample_rate == 16000 else 32
        n_frames = -(-len(audio) // window)
        padded = np.zeros(n_frames * window, dtype=np.float32)
        padded[: len(audio)] = audio
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_size), dtype=np.float32)
        sr = np.array(sample_rate, dtype=np.int64)
        probs = np.empty(n_frames, dtype=np.float32)
        with self._lock:
            for i in range(n_frames):
                x = np.concatenate([context, padded[None, i * window:(i + 1) * window]], axis=1)
                out, state = self.session.run(None, {"input": x, "state": state, "sr": sr})
                probs[i] = out[0, 0]
                context = x[:, -context_size:]
        return probs

    def segments(self, audio: np.ndarray, sample_rate: int, params: VadParams) -> np.ndarray:
        """Speech segments as an (N, 2) float32 array of [start, end] seconds at the input rate."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        x, sr, step = normalize_rate(audio, sample_rate)
        if len(x) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        probs = self.speech_probs(x, sr)
        spans = speech_segments(probs, sr, len(x), params)
        return np.array([[a / sr, b / sr] for a, b in spans], dtype=np.float32).reshape(-1, 2)
