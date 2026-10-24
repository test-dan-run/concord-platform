"""Small, dependency-free metrics for the verifier's sanity checks on the check clip.

They decide pass / warn / fail on one 10 s clip; they are not benchmark scores (Concord Bench scores
with jiwer and pyannote.metrics). SI-SDR judges speech enhancement against the clean clip. DER and VAD error are frame based (10 ms) with Concord's collar
convention: `collar` seconds forgiven on each side of every reference boundary.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

FRAME = 0.01
COLLAR = 0.25
_WS = re.compile(r"\s+")
_SPLITTERS = re.compile(r"[\-‐‑‒–—―/\\_]+")


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def normalise(text: str) -> str:
    """Concord's `basic` normaliser: lowercase, dashes and slashes to spaces, punctuation removed."""
    text = _SPLITTERS.sub(" ", unicodedata.normalize("NFKC", text).lower())
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    return _WS.sub(" ", text).strip()


def wer(ref: str, hyp: str) -> float:
    """Word error rate of normalised strings; 1.0 for an empty reference with any hypothesis."""
    r, h = normalise(ref).split(), normalise(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def _activity(turns: list[Turn], n: int) -> tuple[list[str], np.ndarray]:
    names = sorted({t.speaker for t in turns})
    act = np.zeros((n, max(1, len(names))), dtype=bool)
    for t in turns:
        a, b = max(0, int(round(t.start / FRAME))), min(n, int(round(t.end / FRAME)))
        act[a:b, names.index(t.speaker)] = True
    return names, act


def _scored(ref: list[Turn], n: int, collar: float) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    c = int(round(collar / FRAME))
    for t in ref:
        for b in (t.start, t.end):
            k = int(round(b / FRAME))
            keep[max(0, k - c):min(n, k + c)] = False
    return keep


def der(ref: list[Turn], hyp: list[Turn], duration: float, collar: float = COLLAR) -> float | None:
    """(missed + false alarm + confusion) / reference speech, with the best speaker mapping."""
    n = int(np.ceil(duration / FRAME)) + 1
    _, ra = _activity(ref, n)
    hn, ha = _activity(hyp, n)
    keep = _scored(ref, n, collar)
    ra, ha = ra[keep], ha[keep]
    total = ra.sum()
    if not total:
        return None
    nr, nh = ra.shape[1], (ha.shape[1] if hyp else 0)
    overlap = np.array([[int((ra[:, i] & ha[:, j]).sum()) for j in range(nh)] for i in range(nr)]).reshape(nr, nh)
    # Optimal one-to-one mapping by brute force over the smaller side: the check clip has two speakers.
    best = 0
    if nh and nr <= 7 and nh <= 7:
        if nh >= nr:
            best = max(sum(overlap[i, j] for i, j in enumerate(perm)) for perm in itertools.permutations(range(nh), nr))
        else:
            best = max(sum(overlap[i, j] for j, i in enumerate(perm)) for perm in itertools.permutations(range(nr), nh))
    elif nh:
        best = int(overlap.max(axis=1).sum())  # many speakers: greedy is close enough for a sanity check
    nref, nhyp = ra.sum(1), ha.sum(1)
    missed = np.maximum(nref - nhyp, 0).sum()
    fa = np.maximum(nhyp - nref, 0).sum()
    confusion = np.minimum(nref, nhyp).sum() - best
    return float((missed + fa + confusion) / total)


def vad_errors(ref: list[Turn], hyp: list[tuple[float, float]], duration: float,
               collar: float = COLLAR) -> tuple[float, float]:
    """(missed speech, false-alarm speech) in seconds, speaker-agnostic."""
    n = int(np.ceil(duration / FRAME)) + 1
    _, ra = _activity([Turn(t.start, t.end, "s") for t in ref], n)
    _, ha = _activity([Turn(s, e, "s") for s, e in hyp], n)
    keep = _scored(ref, n, collar)
    r, h = ra[keep, 0], ha[keep, 0]
    return float((r & ~h).sum() * FRAME), float((h & ~r).sum() * FRAME)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio of est against ref, in dB (both zero-mean first)."""
    n = min(len(ref), len(est))
    r = np.asarray(ref[:n], dtype=np.float64)
    e = np.asarray(est[:n], dtype=np.float64)
    r, e = r - r.mean(), e - e.mean()
    energy = float(np.dot(r, r))
    if energy == 0.0:
        return float("-inf")
    target = (np.dot(e, r) / energy) * r
    noise = e - target
    t, d = float(np.dot(target, target)), float(np.dot(noise, noise))
    if d == 0.0:
        return float("inf")
    if t == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(t / d))
