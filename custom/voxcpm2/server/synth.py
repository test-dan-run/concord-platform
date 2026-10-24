"""VoxCPM2 cloning around the `voxcpm` package (2.0.3).

`VoxCPM.generate` takes file paths for the prompt and reference audio, so the reference is written to a temporary
16-bit WAV per request. The denoiser (ModelScope ZipEnhancer) is never loaded: references arrive clean. torch.compile
is on when VOXCPM_OPTIMIZE=1, which the image sets: about 5x faster on the 5090 (RTF 1.17 -> 0.22) after a ~80 s
compile at start-up. VOXCPM_OPTIMIZE=0 turns it off.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODES = ("ultimate", "continuation", "reference")
MIN_REF_S = 0.25
MAX_REF_S = 30.0
MAX_TEXT_CHARS = 2000


@dataclass(frozen=True)
class SynthParams:
    mode: str = "ultimate"
    cfg_value: float = 2.0
    inference_timesteps: int = 10

    @classmethod
    def from_parameters(cls, p: dict) -> SynthParams:
        unknown = sorted(set(p) - {"mode", "cfg_value", "inference_timesteps"})
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; supported: mode, cfg_value, inference_timesteps")
        mode = str(p.get("mode", cls.mode))
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        cfg = float(p.get("cfg_value", cls.cfg_value))
        steps = int(p.get("inference_timesteps", cls.inference_timesteps))
        if not 0.0 < cfg <= 10.0:
            raise ValueError(f"cfg_value must be in (0, 10], got {cfg}")
        if not 1 <= steps <= 100:
            raise ValueError(f"inference_timesteps must be in [1, 100], got {steps}")
        return cls(mode, cfg, steps)


def check_request(text: str, ref: np.ndarray, ref_sr: int, ref_text: str, params: SynthParams) -> None:
    if not text.strip():
        raise ValueError("TEXT is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"TEXT has {len(text)} characters; the limit is {MAX_TEXT_CHARS} (split long turns)")
    if ref_sr < 8000:
        raise ValueError(f"REF_SAMPLE_RATE must be at least 8000, got {ref_sr}")
    dur = len(ref) / ref_sr
    if not MIN_REF_S <= dur <= MAX_REF_S:
        raise ValueError(f"REF_AUDIO is {dur:.2f} s; it must be {MIN_REF_S}-{MAX_REF_S} s")
    if not np.all(np.isfinite(ref)):
        raise ValueError("REF_AUDIO has NaN or inf values")
    if params.mode != "reference" and not ref_text.strip():
        raise ValueError(f"REF_TEXT is required in mode {params.mode!r}")


class VoxCPM2Synth:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.sample_rate = 48000
        self._torch = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from voxcpm import VoxCPM

        if not (self.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"{self.weights_dir / 'model.safetensors'} missing; run `just fetch-weights voxcpm2`")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
        self._torch = torch
        self.model = VoxCPM(voxcpm_model_path=str(self.weights_dir), zipenhancer_model_path=None, enable_denoiser=False,
                            optimize=os.environ.get("VOXCPM_OPTIMIZE") == "1", device=device)
        self.sample_rate = int(self.model.tts_model.sample_rate)

    def _seed(self, seed: int) -> None:
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

    def synthesize(self, text: str, ref: np.ndarray, ref_sr: int, ref_text: str, seed: int,
                   params: SynthParams = SynthParams()) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        check_request(text, ref, ref_sr, ref_text, params)
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ref.wav")
            sf.write(path, np.clip(ref, -1.0, 1.0), ref_sr, subtype="PCM_16")
            kwargs = {}
            if params.mode in ("ultimate", "continuation"):
                kwargs.update(prompt_wav_path=path, prompt_text=ref_text.strip())
            if params.mode in ("ultimate", "reference"):
                kwargs["reference_wav_path"] = path
            with self._lock:
                self._seed(seed)
                wav = self.model.generate(text=text.strip(), cfg_value=params.cfg_value,
                                          inference_timesteps=params.inference_timesteps, **kwargs)
        return np.asarray(wav, dtype=np.float32).reshape(-1), self.sample_rate
