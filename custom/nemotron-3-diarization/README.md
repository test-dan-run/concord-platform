# nemotron-3-diarization

Speaker diarization with [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(Streaming Sortformer, up to 8 speakers) via NeMo, **on CPU**. Contract **oip-v2**, the same
signature as `pyannote-diarization`; alias `default-diarization`. See [WHY.md](WHY.md).

| | |
|---|---|
| Inputs | `AUDIO` FP32 `[-1]` mono; `SAMPLE_RATE` INT32 `[1]`, **16000 only** (resample first) |
| Outputs | `SEGMENTS` FP32 `[-1, 2]` sorted by start; `SPEAKERS` INT32 `[-1]` in arrival order (Sortformer's native ordering) |
| Parameters | none |
| Config | the model card's offline preset (30.4 s input buffer: `chunk_len=340`, `chunk_right_context=40`, `fifo_len=40`, `spkcache_len=264`, `spkcache_update_period=300`), best for whole-file requests |
| Speed | 23 s of two-speaker audio in 0.5 s on CPU (2026-10-17), so it uses no VRAM |
| Weights | existing `${AI_ASSETS}/nemotron-3-diarization/Nemotron-3-Diarization.nemo` |

```python
import numpy as np
from aicore import client
d = client("default-diarization")
out = d.infer({"AUDIO": audio16k.astype(np.float32), "SAMPLE_RATE": np.array([16000], np.int32)})
out["SEGMENTS"], out["SPEAKERS"]
```

**NeMo version.** The model (released 2026-09-23) uses rope self-attention, which the
`nemo-toolkit` 3.0.0 release rejects. `requirements.txt` therefore installs NeMo Speech from a pinned
`main` commit tarball, with `constraints.txt` freezing the rest of the tested environment. Switch to
the PyPI release once a version after 3.0.0 ships.

**Moving it to the GPU.** Add `gpu: {vram_gb: 3}` to the manifest, set `build.base: py-cuda`, and drop
the `+cpu` torch lines from `requirements.txt`. The server already picks CUDA when it is available.
