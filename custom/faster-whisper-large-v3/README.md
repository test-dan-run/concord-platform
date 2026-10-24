# faster-whisper-large-v3

Whisper large-v3 on [CTranslate2](https://github.com/OpenNMT/CTranslate2) via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Contract **openai** /
`audio.transcriptions`, with **word-level timestamps**, which the vLLM `whisper-large-v3` service
does not return. Custom because no pinned engine loads CTranslate2; see [WHY.md](WHY.md).

| | |
|---|---|
| Request | multipart `file` (any format FFmpeg/PyAV decodes), `model`, `language?`, `prompt?`, `temperature?`, `response_format` (`json`, `text`, `verbose_json`), `timestamp_granularities[]` (`segment`, `word`) |
| verbose_json | `text`, `language` (code), `duration`, `segments[]` (OpenAI fields incl. `no_speech_prob`); plus `words[]` (`word`, `start`, `end`, `probability`) when `word` is requested |
| Decoding | beam 5, Silero VAD filter on (skips silence, avoids hallucinations on non-speech), batched windows (below) |
| Batching | faster-whisper's `BatchedInferencePipeline`. VAD cuts the speech into windows of up to 30 s, and the windows of one file are decoded together, `AISVC_MAX_BATCH` at a time (8 when unset). `AISVC_MAX_BATCH_LEN` counts every window as 30 s. When a batch runs out of memory, the server halves it, tries again and keeps the smaller size. Requests run one at a time |
| GPU | GPU by default: float16 (CTranslate2 disables INT8 on sm_120); `vram_gb: 5` (measured 4.1 GB loaded, 4.25 GB peak). Falls back to int8 on CPU when no GPU is visible or the CUDA load/warmup fails (logged on stderr). `ASR_DEVICE=cpu`/`cuda` forces a device, with no fallback |
| Image | slim CPU base + CUDA 12 cuBLAS/runtime **wheels** (`LD_LIBRARY_PATH`), no CUDA image: about 1.5 GB |
| Weights | existing `${AI_ASSETS}/faster-whisper-large-v3` (Systran @edaa852); `weights.yaml` restores it elsewhere |

```python
from aicore import client, model_name
asr = client("faster-whisper-large-v3")
with open("meeting.wav", "rb") as f:
    r = asr.audio.transcriptions.create(model=model_name("faster-whisper-large-v3"), file=f,
                                        response_format="verbose_json", timestamp_granularities=["word"])
r.words[0]  # {'word': ' Hello', 'start': 0.12, 'end': 0.4, 'probability': 0.98}
```

Combine with `default-diarization` (`just up faster-whisper-large-v3 default-diarization`) to label words by speaker: assign
each word to the diarization segment that overlaps its midpoint.

Since 0.2.0 the output differs a little from the sequential decoder of 0.1.0. A window does not see the
text of the window before it, so a name spelled one way early in a file can come out another way later.
There is no retry at a higher temperature when a window repeats itself, and the `prompt` is given to
every window instead of only the first. Segments still follow Whisper's own timestamps, and word
timestamps work as before. A short file of one window runs at the same speed as before; the gain is
on long files.

Pins (`requirements.txt`): `av==18.1.0`, because PyAV 19 removed an argument faster-whisper 1.2.1
passes (fixed upstream, unreleased). Tests mock the model; `test_real_model` and
`test_real_model_batched_matches_sequential` run when faster-whisper and the weights are present
(the first was verified on the RTX 5090 on 2026-10-17).
