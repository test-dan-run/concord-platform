"""DeepFilterNet3 through the official `deep-filter` CLI (Rust + tract, CPU).

The model runs at 48 kHz: input is resampled to 48 kHz, enhanced with delay compensation, resampled back to
the input rate and trimmed or zero-padded to the input length, so timing is preserved.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_FILE = "DeepFilterNet3_onnx.tar.gz"
BINARY = "deep-filter"
MODEL_SR = 48000


@dataclass(frozen=True)
class DfParams:
    atten_lim_db: float = 100.0  # 100 = full noise reduction; lower keeps some of the noisy signal
    post_filter: bool = False

    @classmethod
    def from_parameters(cls, params: dict) -> DfParams:
        unknown = sorted(set(params) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; supported: {sorted(cls.__dataclass_fields__)}")
        p = cls(**{k: params[k] for k in cls.__dataclass_fields__ if k in params})
        if not 0 <= float(p.atten_lim_db) <= 100:
            raise ValueError("atten_lim_db must be in [0, 100]")
        return p


class DeepFilter:
    def __init__(self, weights_dir: str | os.PathLike):
        self.weights_dir = Path(weights_dir)
        self.binary: Path | None = None
        self.model: Path | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        src, model = self.weights_dir / BINARY, self.weights_dir / MODEL_FILE
        for f in (src, model):
            if not f.is_file():
                raise FileNotFoundError(f"{f} missing; run `just fetch-weights deepfilternet3`")
        # the weights mount is read-only and fetched files are not executable: run a private copy
        run_dir = Path(tempfile.mkdtemp(prefix="deepfilter-"))
        binary = run_dir / BINARY
        shutil.copyfile(src, binary)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        subprocess.run([str(binary), "--version"], check=True, capture_output=True, timeout=30)
        self.binary, self.model = binary, model

    def enhance(self, audio: np.ndarray, sample_rate: int, p: DfParams) -> np.ndarray:
        import soundfile as sf
        import soxr

        if self.binary is None:
            raise RuntimeError("model not loaded")
        if sample_rate < 8000:
            raise ValueError(f"SAMPLE_RATE must be at least 8000, got {sample_rate}")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return x
        x48 = soxr.resample(x, sample_rate, MODEL_SR) if sample_rate != MODEL_SR else x
        with tempfile.TemporaryDirectory(prefix="df-") as tmp:
            src, out = Path(tmp) / "in.wav", Path(tmp) / "out"
            sf.write(src, x48, MODEL_SR, subtype="FLOAT")
            cmd = [str(self.binary), "-m", str(self.model), "-D", "-a", str(float(p.atten_lim_db)), "-o", str(out), str(src)]
            if p.post_filter:
                cmd.insert(1, "--pf")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                raise RuntimeError(f"deep-filter failed ({res.returncode}): {res.stderr[-400:]}")
            y48, _ = sf.read(out / "in.wav", dtype="float32", always_2d=True)
        y = y48.mean(axis=1)
        if sample_rate != MODEL_SR:
            y = soxr.resample(y, MODEL_SR, sample_rate)
        y = y[: len(x)]
        if len(y) < len(x):
            y = np.concatenate([y, np.zeros(len(x) - len(y), dtype=np.float32)])
        return y.astype(np.float32)
