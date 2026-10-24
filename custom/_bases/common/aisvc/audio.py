"""Audio encoding for audio.speech responses (OpenAI `response_format`)."""

from __future__ import annotations

import io
import wave

import numpy as np

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# soundfile (libsndfile) format/subtype per response_format; aac has no libsndfile encoder.
_SOUNDFILE = {"mp3": ("MP3", "MPEG_LAYER_III"), "flac": ("FLAC", "PCM_16"), "opus": ("OGG", "OPUS")}


def to_int16(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def encode_audio(samples: np.ndarray, sample_rate: int, response_format: str) -> bytes:
    """Encode mono float samples in [-1, 1]. Raises ValueError for unsupported formats."""
    if response_format == "pcm":  # raw 16-bit signed little-endian, like OpenAI's pcm
        return to_int16(samples).tobytes()
    if response_format == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(to_int16(samples).tobytes())
        return buf.getvalue()
    if response_format in _SOUNDFILE:
        try:
            import soundfile as sf
        except ImportError:
            raise ValueError(f"response_format {response_format!r} needs the soundfile package; use 'wav' or 'pcm'") from None
        fmt, subtype = _SOUNDFILE[response_format]
        buf = io.BytesIO()
        sf.write(buf, np.asarray(samples, dtype=np.float32).reshape(-1), sample_rate, format=fmt, subtype=subtype)
        return buf.getvalue()
    raise ValueError(f"response_format {response_format!r} is not supported by this service (use wav, pcm, flac, mp3 or opus)")
