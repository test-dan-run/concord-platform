"""Speaker turns -> the shared diarization tensors (identical in every diarization service)."""

from __future__ import annotations

from collections.abc import Iterable, Hashable

import numpy as np


def turns_to_tensors(turns: Iterable[tuple[float, float, Hashable]]) -> dict[str, np.ndarray]:
    """SEGMENTS (N, 2) start/end seconds sorted by start; SPEAKERS (N,) indices in order of first appearance."""
    ordered = sorted(turns, key=lambda t: (t[0], t[1]))
    index: dict[Hashable, int] = {}
    for _, _, label in ordered:
        index.setdefault(label, len(index))
    segments = np.array([[s, e] for s, e, _ in ordered], dtype=np.float32).reshape(-1, 2)
    speakers = np.array([index[label] for _, _, label in ordered], dtype=np.int32)
    return {"SEGMENTS": segments, "SPEAKERS": speakers}
