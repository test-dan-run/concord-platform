# pyannote-diarization

Speaker diarization with [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
on the GPU. Contract **oip-v2** (audio in, no OpenAI route); see [WHY.md](WHY.md).
It shares a signature with `nemotron-3-diarization`, so apps can switch between them.

| | |
|---|---|
| Inputs | `AUDIO` FP32 `[-1, -1]` mono recordings in [-1, 1], padded with zeros to one length; `LENGTHS` INT32 `[-1]`; `SAMPLE_RATE` INT32 `[1]`, 8000 to 48000 for every recording (pyannote resamples to 16 kHz) |
| Outputs | `SEGMENTS` FP32 `[-1, 2]` start and end seconds; `SPEAKERS` INT32 `[-1]`, the speaker of each segment; `ROWS` INT32 `[-1]`, the recording of each segment. The segments come recording after recording, each recording's sorted by start, and speakers are numbered per recording in order of first appearance (0 is the first voice heard) |
| One recording | `AUDIO` FP32 `[-1]` without `LENGTHS` still works, and gets `SEGMENTS` and `SPEAKERS` without `ROWS`, as before |
| Parameters | `num_speakers`, `min_speakers`, `max_speakers` (hints), `exclusive` (true: no overlapping turns, easier to align with ASR words), for every recording in the request |
| Batch settings | `PYANNOTE_SEGMENTATION_BATCH` and `PYANNOTE_EMBEDDING_BATCH`, default 32 each from the 3.1 config (see below) |
| GPU | `vram_gb: 3`; base `py-cuda-torch211` (pyannote.audio 4 needs torchaudio, last released for torch 2.11) |
| Weights | existing `${AI_ASSETS}/pyannote-speaker-diarization-3.1` (`config.yaml`, `segmentation-3.0/`, `wespeaker-voxceleb-resnet34-LM/`). Two source repos are gated, so `just fetch-weights` needs `HF_TOKEN` with their terms accepted |

```python
import numpy as np
from aicore import client
d = client("pyannote-diarization")
out = d.infer({"AUDIO": audio.astype(np.float32), "SAMPLE_RATE": np.array([16000], np.int32)},  # one recording
              parameters={"max_speakers": 4, "exclusive": True})
list(zip(out["SEGMENTS"].tolist(), out["SPEAKERS"].tolist()))  # [([0.33, 7.84], 0), ([8.69, 16.53], 1), ...]
```

Implementation notes (`server/diarizer.py`):

- **Batches.** The pipeline already batches inside each recording. It runs the segmentation model on 10 s
  windows at a 1 s step, `PYANNOTE_SEGMENTATION_BATCH` windows at a time, and the embedding model on every
  (window, speaker) pair, `PYANNOTE_EMBEDDING_BATCH` pairs at a time. GPU memory depends on those two numbers
  and not on the length of the recording, so a long recording already fills the batches. The recordings of
  one request run one after another. On a CUDA out of memory error both numbers are halved for the rest of
  the process and the recording runs again. `AISVC_MAX_BATCH` is not used, because it counts rows.
- **Fully offline.** The pipeline is built from the local checkpoints instead of the config's Hub ids.
- **PLDA placeholder.** pyannote.audio 4 always loads a PLDA model, by default from the gated
  community-1 repo, although only `VBxClustering` uses it. 3.1 uses `AgglomerativeClustering`, so an
  inert placeholder is passed.
- **No FFmpeg/torchcodec.** Audio arrives in memory, so they are never used (verified with torchcodec removed).

Tested 2026-10-17 on CPU (torch 2.11) on a two-speaker clip at 16 kHz and 8 kHz. The GPU image has not
been built yet.
