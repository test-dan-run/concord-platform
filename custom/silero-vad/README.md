# silero-vad

Voice activity detection with [Silero VAD](https://github.com/snakers4/silero-vad) v6.2.3 (ONNX,
CPU). Contract **oip-v2** (non-text input, no OpenAI route). See [WHY.md](WHY.md).

| | |
|---|---|
| Inputs | `AUDIO` FP32 `[-1]` mono samples in [-1, 1]; `SAMPLE_RATE` INT32 `[1]`: 8000, 16000 or a multiple of 16000 |
| Output | `SEGMENTS` FP32 `[-1, 2]`: `[start, end]` in seconds |
| Parameters | `threshold` (0.5), `neg_threshold` (threshold-0.15), `min_speech_duration_ms` (250), `min_silence_duration_ms` (100), `speech_pad_ms` (30) |
| Weights | `weights.yaml` → `${AI_ASSETS}/custom-models/silero-vad/v6.2.3/silero_vad.onnx` |

```bash
just fetch-weights silero-vad
just build silero-vad
just up utils
just conform silero-vad
```

```python
import numpy as np
from aicore import client
vad = client("default-vad")
out = vad.infer({"AUDIO": audio.astype(np.float32), "SAMPLE_RATE": np.array([16000], dtype=np.int32)},
                parameters={"threshold": 0.6})
out["SEGMENTS"]  # array([[0.38, 2.1], ...], dtype=float32)
```

Code: `server/vad.py` ports Silero's `OnnxWrapper` and `get_speech_timestamps` to numpy (no torch
in the image); `server/app.py` is only the handler on top of `aisvc.create_oip_app`.
Tests (`tests/`) mock the model and run the shared oip-v2 conformance suite in-process. When the
ONNX file is on disk, a real-model test also runs.
