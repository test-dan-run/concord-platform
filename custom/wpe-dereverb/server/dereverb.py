"""Weighted prediction error (WPE) dereverberation, single channel, with nara_wpe (non-AI, statistical).

STFT 512 / shift 128 (32 ms / 8 ms at 16 kHz), prediction filter of `taps` frames after a `delay` of frames,
`iterations` of the WPE update. Output keeps the input rate and length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WpeParams:
    taps: int = 10
    delay: int = 3
    iterations: int = 3

    @classmethod
    def from_parameters(cls, params: dict) -> WpeParams:
        unknown = sorted(set(params) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; supported: {sorted(cls.__dataclass_fields__)}")
        given = {k: params[k] for k in cls.__dataclass_fields__ if k in params}
        for k, v in given.items():
            if isinstance(v, bool) or not isinstance(v, int):  # a fraction is refused, not cut to an integer
                raise ValueError(f"{k} must be an integer")
        p = cls(**given)
        if not (1 <= p.taps <= 40 and 1 <= p.delay <= 10 and 1 <= p.iterations <= 10):
            raise ValueError("taps must be 1-40, delay 1-10, iterations 1-10")
        return p


def stft_size(sample_rate: int) -> tuple[int, int]:
    """32 ms windows, 8 ms shift, as a power of two at the given rate."""
    size = 1 << int(round(np.log2(0.032 * sample_rate)))
    return size, size // 4


def dereverb(audio: np.ndarray, sample_rate: int, p: WpeParams) -> np.ndarray:
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    if sample_rate < 8000:
        raise ValueError(f"SAMPLE_RATE must be at least 8000, got {sample_rate}")
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    size, shift = stft_size(sample_rate)
    if len(y) < 4 * size:
        return y.astype(np.float32)  # too short for a filter estimate; returned unchanged
    Y = stft(y[None, :], size=size, shift=shift).transpose(2, 0, 1)  # (freq, channel, frame)
    Z = wpe(Y, taps=p.taps, delay=p.delay, iterations=p.iterations, statistics_mode="full").transpose(1, 2, 0)
    z = istft(Z, size=size, shift=shift)[0]
    z = z[: len(y)]
    if len(z) < len(y):
        z = np.concatenate([z, np.zeros(len(y) - len(z))])
    return z.astype(np.float32)
