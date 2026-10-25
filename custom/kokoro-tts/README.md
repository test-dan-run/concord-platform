# kokoro-tts

Text-to-speech with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the GPU.
Contract **openai** / `audio.speech`: `POST /v1/audio/speech`, same request and response as OpenAI.
Custom because no pinned engine can serve it; see [WHY.md](WHY.md).

| | |
|---|---|
| Request | `model` (`kokoro-tts`), `input` (≤ 4096 chars), `voice`, `response_format` (`wav`, `pcm`, `flac`, `mp3`, `opus`; `aac` unsupported), `speed` 0.25–4.0 |
| Voices | OpenAI names (`alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`, `coral`, `sage`, `ash`, `ballad`, `verse`) map to Kokoro voices; Kokoro ids (`af_heart`, `bm_george`, ...) work directly. English (American `a*`, British `b*`) only. |
| Output | 24 kHz mono; `pcm` is raw 16-bit little-endian like OpenAI's |
| GPU | `vram_gb: 3`; `aisvc.gpu.limit_torch_memory` caps torch at that budget |
| Weights | `weights.yaml` → `${AI_ASSETS}/custom-models/kokoro-tts/v1.0/{config.json,kokoro-v1_0.pth,voices/*.pt}` (~340 MB) |

```bash
just fetch-weights kokoro-tts
just build kokoro-tts        # builds py-cuda-base first if needed (torch 2.14.1+cu130, sm_120)
just up kokoro-tts
just conform kokoro-tts
```

```python
from aicore import client, model_name
tts = client("default-tts")
tts.audio.speech.create(model=model_name("default-tts"), voice="af_heart",
                        input="Hello there!", response_format="wav").write_to_file("hello.wav")
```

The image is fully offline at runtime (`HF_HUB_OFFLINE=1`). The spaCy English model is installed
at build time, and model, config and voices load from `/models`.
Tests (`tests/`) mock the model, so they need no torch, weights or GPU.
