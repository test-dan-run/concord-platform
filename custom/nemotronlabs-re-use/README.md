# nemotronlabs-re-use

Universal speech enhancement with [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)
(OpenMDW-1.1): one model for additive noise, reverberation, clipping, bandwidth limits, codec artifacts, packet
loss and poor microphones, at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz, in any language. With overlapping speech it
keeps the main speaker and suppresses the others. GPU only.

```python
import numpy as np
from aicore import client

clips = [clip_a, clip_b]  # float32 mono clips at one rate
audio = np.zeros((len(clips), max(map(len, clips))), np.float32)
for row, c in zip(audio, clips):
    row[: len(c)] = c
out = client("nemotronlabs-re-use").infer({"AUDIO": audio, "LENGTHS": np.array([len(c) for c in clips], np.int32),
                                           "SAMPLE_RATE": np.array([16000], dtype=np.int32)})
enhanced = [row[:n] for row, n in zip(out["AUDIO"], out["LENGTHS"])]  # each at the rate and length of its input
```

- **Input:** `AUDIO` FP32 `[-1, -1]`, mono clips in -1 to 1 padded with zeros to one length; `LENGTHS` INT32
  `[-1]`, the length of each clip; `SAMPLE_RATE` INT32 `[1]`, one of the rates above for every clip (others are a 400).
- **Output:** `AUDIO` FP32 `[-1, -1]` padded with zeros, `LENGTHS` and `SAMPLE_RATE`. Each row keeps its input's
  rate and length. The signature is the same as `deepfilternet3` and `wpe-dereverb`.
- **One clip:** a request with `AUDIO` FP32 `[-1]` and no `LENGTHS` still works, and it gets `AUDIO` FP32 `[-1]`
  and `SAMPLE_RATE` back, as before the batch signature.
- **Batches:** each clip is enhanced whole, or in 5 s chunks when it is long (see below). Pieces of the same
  length share one pass on the GPU, so the chunks of long files run together, and so do clips of equal length.
  Clips of different lengths run one after another, because padding would change the output (`WHY.md`).
  `AISVC_MAX_BATCH` and `AISVC_MAX_BATCH_LEN` limit a pass; the length unit is samples / 16 (milliseconds
  at 16 kHz), because memory grows with samples at any rate.
- **Parameters:** none.
- **Long audio:** up to 240,000 samples (15 s at 16 kHz, 10 s at 24 kHz, 5 s at 48 kHz) is enhanced in one pass; longer
  inputs in 5 s chunks with 50% Hann overlap-add, as the model card advises (training clips were under 10 s).
  The limit keeps GPU memory under about 3 GiB at every rate. Tensors travel as JSON, so send files of minutes in
  pieces from the caller.
- **Weights:** `just fetch-weights nemotronlabs-re-use` (39 MB, sha256-pinned) into
  `${AI_ASSETS}/custom-models/nemotronlabs-re-use/31f82f2`.
- **Build:** `just build nemotronlabs-re-use` compiles mamba-ssm's kernel for sm_120 on the first build (see `WHY.md`).
