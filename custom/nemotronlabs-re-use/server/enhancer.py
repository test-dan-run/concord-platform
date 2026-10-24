"""NVIDIA NemotronLabs RE-USE on in-memory audio: the steps of the model repo's inference.py and
inference_chunk.py (revision 31f82f2), without files, torchaudio or Hub access.

The STFT sizes scale with the input rate (the model was configured at 8 kHz: n_fft 320, hop 40), so
every supported rate is enhanced at its own rate and the output keeps the input's rate and length.
Inputs over WHOLE_MAX_SAMPLES are enhanced in 5 s chunks with 50% Hann overlap-add, as the model card
advises for long files (its training clips are under 10 s). The limit is in samples because memory grows
with the rate: one 10 s pass at 48 kHz peaked at 5.6 GiB on the RTX 5090, a 5 s one under 3 GiB.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

SUPPORTED_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
WHOLE_MAX_SAMPLES = 240_000  # one pass up to 10 s at 24 kHz (5 s at 48 kHz); beyond, chunks
CHUNK_S = 5.0
HOP_PORTION = 0.5
FILES = ("config.json", "model.safetensors")


def _even(value: float) -> int:
    v = int(round(value))
    return v if v % 2 == 0 else v + 1


class ReUse:
    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.model = None
        self.cfg: dict | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from aisvc.gpu import limit_torch_memory
        from safetensors.torch import load_file

        from .reuse.generator_SEMamba_time_d4 import SEMamba

        if not torch.cuda.is_available():
            raise RuntimeError("RE-USE needs a CUDA GPU: its Mamba layers run on mamba-ssm kernels (no CPU mode)")
        for f in FILES:
            if not (self.dir / f).is_file():
                raise FileNotFoundError(f"{self.dir / f} missing; run `just fetch-weights nemotronlabs-re-use`")
        self.cfg = json.loads((self.dir / "config.json").read_text())
        limit_torch_memory(torch)
        model = SEMamba(self.cfg)
        model.load_state_dict(load_file(str(self.dir / "model.safetensors")), strict=True)
        self.model = model.to(torch.device("cuda")).eval()

    def _sizes(self, sample_rate: int) -> tuple[int, int, int]:
        assert self.cfg is not None
        s = self.cfg["stft_cfg"]
        base = s["sampling_rate"]  # inference.py: make_even(size * noisy_sr // sampling_rate)
        return tuple(_even(s[k] * sample_rate // base) for k in ("n_fft", "hop_size", "win_size"))  # type: ignore[return-value]

    def _enhance_one(self, wav, sample_rate: int):
        """One pass of the model over a [1, N] tensor (inference.py)."""
        import torch

        from .reuse.stfts import mag_phase_istft, mag_phase_stft

        n_fft, hop, win = self._sizes(sample_rate)
        compress = self.cfg["model_cfg"]["compress_factor"]
        n = wav.shape[1]
        if n <= n_fft:  # the STFT's reflect padding needs more than n_fft / 2 samples
            wav = torch.nn.functional.pad(wav, (0, n_fft + 1 - n))
        mag, pha, _ = mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win, compress_factor=compress,
                                     center=True, addeps=False)
        amp_g, pha_g, _ = self.model(mag, pha)
        # inference.py: zero frames that are mostly zero ("to remove strange sweep artifact")
        zero_portion = torch.sum(torch.expm1(torch.relu(amp_g)) == 0, 1) / amp_g.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress)
        if out.shape[1] >= n:  # pad_or_trim_to_match
            return out[:, :n]
        return torch.nn.functional.pad(out, (0, n - out.shape[1]), value=1e-8)

    def _enhance_chunked(self, wav, sample_rate: int):
        """Hann-windowed overlap-add over CHUNK_S chunks (inference_chunk.py)."""
        import math

        import torch

        chunk = int(CHUNK_S * sample_rate)
        hop = int(HOP_PORTION * chunk)
        window = torch.clamp(torch.hann_window(chunk, device=wav.device), min=1e-7)
        out = torch.zeros_like(wav)
        weight = torch.zeros_like(wav)
        n_chunks = max(1, math.ceil((wav.shape[1] - chunk) / hop) + 1)
        for i in range(n_chunks):
            sl = slice(i * hop, i * hop + chunk) if i < n_chunks - 1 else slice(wav.shape[1] - chunk, wav.shape[1])
            piece = self._enhance_one(wav[:, sl], sample_rate)
            out[:, sl] += piece * window[: piece.shape[1]]
            weight[:, sl] += window[: piece.shape[1]]
        return out / weight

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call load() first")
        if sample_rate not in SUPPORTED_RATES:
            raise ValueError(f"SAMPLE_RATE {sample_rate} is not supported; use one of {', '.join(map(str, SUPPORTED_RATES))}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("AUDIO is empty")
        if not np.all(np.isfinite(audio)):
            raise ValueError("AUDIO contains NaN or infinite values")
        with self._lock, torch.inference_mode():
            wav = torch.from_numpy(audio).to("cuda").unsqueeze(0)
            if audio.size > WHOLE_MAX_SAMPLES:
                out = self._enhance_chunked(wav, sample_rate)
            else:
                out = self._enhance_one(wav, sample_rate)
            return out[0].float().cpu().numpy()
