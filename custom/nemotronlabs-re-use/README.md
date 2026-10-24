# nemotronlabs-re-use

Universal speech enhancement with [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)
(OpenMDW-1.1): one model for additive noise, reverberation, clipping, bandwidth limits, codec artifacts, packet
loss and poor microphones, at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz, in any language. With overlapping speech it
keeps the main speaker and suppresses the others. GPU only.

```python
import numpy as np
from aicore import client

out = client("nemotronlabs-re-use").infer({"AUDIO": audio_f32_mono, "SAMPLE_RATE": np.array([16000], dtype=np.int32)})
enhanced = out["AUDIO"]          # same rate and length as the input
```

- **Input:** `AUDIO` FP32 `[-1]` mono in −1…1; `SAMPLE_RATE` INT32 `[1]`, one of the rates above (others are a 400).
- **Output:** `AUDIO` FP32 `[-1]` and `SAMPLE_RATE`, unchanged rate and length.
- **Parameters:** none.
- **Long audio:** up to 240,000 samples (10 s at 16 or 24 kHz, 5 s at 48 kHz) is enhanced in one pass; longer
  inputs in 5 s chunks with 50% Hann overlap-add, as the model card advises (training clips were under 10 s).
  The limit keeps GPU memory under about 3 GiB at every rate. Tensors travel as JSON, so send files of minutes in
  pieces from the caller.
- **Weights:** `just fetch-weights nemotronlabs-re-use` (39 MB, sha256-pinned) into
  `${AI_ASSETS}/custom-models/nemotronlabs-re-use/31f82f2`.
- **Build:** `just build nemotronlabs-re-use` compiles mamba-ssm's kernel for sm_120 on the first build (see `WHY.md`).
