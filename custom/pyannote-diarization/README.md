# pyannote-diarization

Speaker diarization with [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
on the GPU. Contract **oip-v2** (audio in, no OpenAI route); see [WHY.md](WHY.md).
It shares a signature with `nemotron-3-diarization`, so apps can switch between them.

| | |
|---|---|
| Inputs | `AUDIO` FP32 `[-1]` mono in [-1, 1]; `SAMPLE_RATE` INT32 `[1]`, 8000–48000 (pyannote resamples to 16 kHz) |
| Outputs | `SEGMENTS` FP32 `[-1, 2]` start/end seconds, sorted by start; `SPEAKERS` INT32 `[-1]`, speaker index per segment in order of first appearance (0 = first voice heard) |
| Parameters | `num_speakers`, `min_speakers`, `max_speakers` (hints), `exclusive` (true: no overlapping turns, easier to align with ASR words) |
| GPU | `vram_gb: 3`; base `py-cuda-torch211` (pyannote.audio 4 needs torchaudio, last released for torch 2.11) |
| Weights | existing `${AI_ASSETS}/pyannote-speaker-diarization-3.1` (`config.yaml`, `segmentation-3.0/`, `wespeaker-voxceleb-resnet34-LM/`). Two source repos are gated, so `just fetch-weights` needs `HF_TOKEN` with their terms accepted |

```python
import numpy as np
from aicore import client
d = client("pyannote-diarization")
out = d.infer({"AUDIO": audio.astype(np.float32), "SAMPLE_RATE": np.array([16000], np.int32)},
              parameters={"max_speakers": 4, "exclusive": True})
list(zip(out["SEGMENTS"].tolist(), out["SPEAKERS"].tolist()))  # [([0.33, 7.84], 0), ([8.69, 16.53], 1), ...]
```

Implementation notes (`server/diarizer.py`):

- **Fully offline.** The pipeline is built from the local checkpoints instead of the config's Hub ids.
- **PLDA placeholder.** pyannote.audio 4 always loads a PLDA model, by default from the gated
  community-1 repo, although only `VBxClustering` uses it. 3.1 uses `AgglomerativeClustering`, so an
  inert placeholder is passed.
- **No FFmpeg/torchcodec.** Audio arrives in memory, so they are never used (verified with torchcodec removed).

Tested 2026-10-17 on CPU (torch 2.11) on a two-speaker clip at 16 kHz and 8 kHz. The GPU image has not
been built yet.
