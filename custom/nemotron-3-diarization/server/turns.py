"""Speaker turns -> the shared diarization tensors (identical in every diarization service; tests/test_images.py checks)."""

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


def stack_rows(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The tensors of a batch: every row's SEGMENTS and SPEAKERS in row order, and ROWS (N,), the row of each
    segment. Speaker indices stay per row, so 0 is the first voice heard in that row."""
    return {"SEGMENTS": np.concatenate([r["SEGMENTS"] for r in results] or [np.zeros((0, 2))]).astype(np.float32),
            "SPEAKERS": np.concatenate([r["SPEAKERS"] for r in results] or [np.zeros(0)]).astype(np.int32),
            "ROWS": np.concatenate([np.full(len(r["SPEAKERS"]), i) for i, r in enumerate(results)] or [np.zeros(0)]).astype(np.int32)}
