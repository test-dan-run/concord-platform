# Platform services

Every service the platform defines, one manifest each in `services/*.yaml`. How to run, add and serve
models is in the platform guide, [`../README.md`](../README.md); the model catalog's contract is in
[`catalog/README.md`](catalog/README.md).

## Services

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| qwen3-8b | default-chat | model | vllm | openai / chat.completions | 8201 | util 0.35 |
| qwen3-32b-awq | heavy-chat | model | vllm | openai / chat.completions | 8202 | util 0.85 |
| whisper-large-v3 | default-asr | model | vllm | openai / audio.transcriptions | 8101 | util 0.15 |
| bge-m3 | default-embed | model | vllm | openai / embeddings | 8301 | util 0.08 |
| kokoro-tts | default-tts | model | custom | openai / audio.speech | 8103 | 3 GB |
| silero-vad | default-vad | model | custom | oip-v2 / infer | 8401 | CPU |
| text-normalizer | default-normalizer | utility | custom | openai-inspired / text.transform | 8501 | CPU |
| qwen3-4b-instruct-2507 | fast-chat | model | vllm | openai / chat.completions | 8203 | util 0.25 |
| qwen2-5-14b-instruct-awq | mid-chat | model | vllm | openai / chat.completions | 8204 | util 0.45 |
| qwen3-5-4b | default-vlm | model | vllm | openai / chat.completions (images, video) | 8205 | util 0.40 |
| faster-whisper-large-v3 | | model | custom | openai / audio.transcriptions (word timestamps) | 8102 | 5 GB |
| pyannote-diarization | | model | custom | oip-v2 / infer | 8402 | 3 GB |
| nemotron-3-diarization | default-diarization | model | custom | oip-v2 / infer | 8403 | CPU |
| whisper-large-v3-turbo | | model | vllm | openai / audio.transcriptions | 8104 | util 0.10 |
| qwen-sea-lion-v4-5-27b-it | | model | vllm | openai / chat.completions | 8206 | util 0.75 |
| gemma-sea-lion-v4-27b-it-nvfp4 | | model | vllm | openai / chat.completions | 8207 | util 0.80 |
| apertus-sea-lion-v4-8b-it | | model | vllm | openai / chat.completions | 8208 | util 0.68 |
| voxcpm2 | | model | custom | oip-v2 / infer | 8410 | 8 GB |
| qwen3-asr-1-7b | | model | vllm | openai / audio.transcriptions | 8106 | util 0.22 |
| meralion-3-3b-asr | | model | custom | openai / audio.transcriptions | 8107 | 9 GB |
| granite-speech-5-0-2b | | model | vllm | openai / audio.transcriptions | 8105 | util 0.24 |
| cohere-transcribe-03-2026 | | model | vllm | openai / audio.transcriptions | 8108 | util 0.22 |
| ark-asr-0-6b | | model | custom | openai / audio.transcriptions | 8109 | 3 GB |
| wavlm-sv | | model | custom | oip-v2 / infer | 8404 | 3 GB |
| redimnet2-sv | | model | custom | oip-v2 / infer | 8405 | 2 GB |
| utmos | | model | custom | oip-v2 / infer | 8406 | CPU |
| qwen3-forced-aligner | | model | custom | oip-v2 / infer | 8407 | 4 GB |
| nemotronlabs-re-use | | model | custom | oip-v2 / infer (speech enhancement) | 8411 | 4 GB |
| monkey-custom | | model | custom | openai-inspired / text.transform (transcript normalisation) | 8412 | CPU |

Service names use kebab case (lowercase words joined by hyphens), so `qwen2.5` and `qwen3.5` become
`qwen2-5` and `qwen3-5`. When an engine model's `model` field points at `/models/...`, the service
uses your existing copy of the weights in `$AI_ASSETS`, which is mounted read-only.

The two diarization services use the same OIP signature, so apps can switch between them without
code changes. Diarization means labeling which speaker talks when in an audio recording. The
signature takes `AUDIO` and `SAMPLE_RATE` as inputs, and it returns `SEGMENTS` and `SPEAKERS`, with
speakers listed in the order they first appear.

Ports are grouped by service type. If a port is outside its group, validation gives a warning
instead of an error.

| Port range | Services |
|---|---|
| `81xx` | Audio models |
| `82xx` | Chat models and vision language models (VLMs) |
| `83xx` | Embedding and reranking models |
| `84xx` | Other models, usually on the OIP contract |
| `85xx` | Utilities |

The model catalog listens on `8700`, outside these bands (Concord Bench, Synthetics and Concord Label
use `8600`, `8610` and `8620`).
