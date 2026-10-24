"""DNSMOS P.835 (non-personalised): SIG, BAK and OVRL mean opinion scores for a speech clip.

Follows microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (commit 6412bc5): 16 kHz input, clips shorter than
9.01 s are repeated to that length, windows of 9.01 s every 1 s are scored and averaged, and the raw outputs
are mapped through the published polynomial fits. The P.808 head (needs a librosa mel front end) is not used.
"""

from __future__ import annotations

import threading
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
        self._lock = threading.Lock()

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
        with self._lock:
            return self.session.run(None, {"input_1": segment.astype(np.float32)[None, :]})[0][0]

    def score(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
        """(OVRL, SIG, BAK) for one clip."""
        import soxr

        if self.session is None:
            raise RuntimeError("model not loaded")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            raise ValueError("empty clip")
        if sample_rate != SR:
            x = soxr.resample(x, sample_rate, SR).astype(np.float32)
        sig, bak, ovr = [], [], []
        for seg in windows(x):
            s, b, o = self.raw(seg)
            sig.append(P_SIG(s))
            bak.append(P_BAK(b))
            ovr.append(P_OVR(o))
        return float(np.mean(ovr)), float(np.mean(sig)), float(np.mean(bak))
