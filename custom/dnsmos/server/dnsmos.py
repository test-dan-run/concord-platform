"""DNSMOS P.835 (non-personalised): SIG, BAK and OVRL mean opinion scores for a speech clip.

Follows microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (commit 6412bc5): 16 kHz input, clips shorter than
9.01 s are repeated to that length, windows of 9.01 s every 1 s are scored and averaged, and the raw outputs
are mapped through the published polynomial fits. The P.808 head (needs a librosa mel front end) is not used.
Every window has the same length, so the windows of a whole batch of clips go through the model in one run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_FILE = "sig_bak_ovr.onnx"
SR = 16000
INPUT_LENGTH = 9.01
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


def windows(audio: np.ndarray) -> list[np.ndarray]:
    len_samples = int(INPUT_LENGTH * SR)
    a = audio
    if len(a) < len_samples:  # the reference appends the clip to itself until it is long enough
        a = np.tile(audio, -(-len_samples // len(audio)))
    num_hops = int(np.floor(len(a) / SR) - INPUT_LENGTH) + 1
    out = []
    for idx in range(num_hops):
        seg = a[int(idx * SR): int((idx + INPUT_LENGTH) * SR)]
        if len(seg) >= len_samples:
            out.append(seg[:len_samples])
    return out


class Dnsmos:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.session = None

    def load(self) -> None:
        import onnxruntime as ort

        path = self.weights_dir / MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run `just fetch-weights dnsmos`")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def raw(self, segment: np.ndarray) -> np.ndarray:
        """(sig, bak, ovr) raw outputs for one 9.01 s window."""
        return self.raw_batch([segment])[0]

    def raw_batch(self, segments: list[np.ndarray]) -> np.ndarray:
        """[N, 3] raw outputs for N windows of 9.01 s, in one run."""
        return self.session.run(None, {"input_1": np.stack(segments).astype(np.float32)})[0]

    def score(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
        """(OVRL, SIG, BAK) for one clip."""
        return self.score_batch([prepare(audio, sample_rate)])[0]

    def score_batch(self, clips: list[np.ndarray]) -> list[tuple[float, float, float]]:
        """(OVRL, SIG, BAK) for each prepared clip (`prepare`). Callers serialise calls (aisvc.batch.MicroBatcher)."""
        if self.session is None:
            raise RuntimeError("model not loaded")
        wins = [windows(x) for x in clips]
        raw = self.raw_batch([w for ws in wins for w in ws])
        out, k = [], 0
        for ws in wins:
            s, b, o = raw[k:k + len(ws)].T
            k += len(ws)
            out.append((float(np.mean(P_OVR(o))), float(np.mean(P_SIG(s))), float(np.mean(P_BAK(b)))))
        return out


def prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        raise ValueError("empty clip")
    if sample_rate != SR:
        import soxr

        x = soxr.resample(x, sample_rate, SR).astype(np.float32)
    return x
