# Platform services

This file lists every service the platform defines, and each service has one manifest in
`services/*.yaml`. The platform guide in [`../README.md`](../README.md) explains how to run, add and
serve models, and [`catalog/README.md`](catalog/README.md) describes the model catalog's API.

## Services by group

The tables below group the services by what they do. The GPU column shows the least GPU memory a
service needs to load, as a share of the GPU for a vLLM service (`util`) or in GB for a custom service,
or `CPU` for a service that doesn't use the GPU. Only one GPU service runs at a time. When it starts, it
gets the GPU memory that is free at that moment less 1 GiB, and at most `util_limit` of the GPU.
`just up` and `just vram` refuse a set that names two or more GPU services.

### Chat and vision language models

Every chat model uses the `openai` contract with the `chat.completions` API, and `qwen3-5-4b` also reads
images and video. A vision language model (VLM) is a chat model that can also read images.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| qwen3-8b | default-chat | model | vllm | openai / chat.completions | 8201 | util 0.35 |
| qwen3-32b-awq | heavy-chat | model | vllm | openai / chat.completions | 8202 | util 0.85 |
| qwen3-4b-instruct-2507 | fast-chat | model | vllm | openai / chat.completions | 8203 | util 0.25 |
| qwen2-5-14b-instruct-awq | mid-chat | model | vllm | openai / chat.completions | 8204 | util 0.45 |
| qwen3-5-4b | default-vlm | model | vllm | openai / chat.completions (images, video) | 8205 | util 0.40 |
| qwen-sea-lion-v4-5-27b-it |  | model | vllm | openai / chat.completions | 8206 | util 0.75 |
| gemma-sea-lion-v4-27b-it-nvfp4 |  | model | vllm | openai / chat.completions | 8207 | util 0.80 |
| apertus-sea-lion-v4-8b-it |  | model | vllm | openai / chat.completions | 8208 | util 0.68 |

### Speech to text (ASR)

Automatic speech recognition (ASR) turns recorded speech into text, and every ASR service uses the
`openai` contract with the `audio.transcriptions` API.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| whisper-large-v3 | default-asr | model | vllm | openai / audio.transcriptions | 8101 | util 0.15 |
| faster-whisper-large-v3 |  | model | custom | openai / audio.transcriptions (word timestamps) | 8102 | 5 GB |
| whisper-large-v3-turbo |  | model | vllm | openai / audio.transcriptions | 8104 | util 0.10 |
| granite-speech-5-0-2b |  | model | vllm | openai / audio.transcriptions | 8105 | util 0.24 |
| qwen3-asr-1-7b |  | model | vllm | openai / audio.transcriptions | 8106 | util 0.22 |
| meralion-3-3b-asr |  | model | custom | openai / audio.transcriptions | 8107 | 9 GB |
| cohere-transcribe-03-2026 |  | model | vllm | openai / audio.transcriptions | 8108 | util 0.22 |
| ark-asr-0-6b |  | model | custom | openai / audio.transcriptions | 8109 | 3 GB |

### Text to speech

Text to speech services turn text into recorded speech, and `voxcpm2` can also copy the voice in a short
sample recording.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| kokoro-tts | default-tts | model | custom | openai / audio.speech | 8103 | 3 GB |
| voxcpm2 |  | model | custom | oip-v2 / infer | 8410 | 8 GB |

### Embeddings

An embedding model turns text into a list of numbers, so that similar texts get similar numbers for
search and retrieval.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| bge-m3 | default-embed | model | vllm | openai / embeddings | 8301 | util 0.08 |

### Voice activity detection and diarization

Voice activity detection (VAD) finds the parts of a recording that contain speech, and diarization
labels which speaker talks when.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| silero-vad | default-vad | model | custom | oip-v2 / infer | 8401 | CPU |
| pyannote-diarization |  | model | custom | oip-v2 / infer | 8402 | 3 GB |
| nemotron-3-diarization | default-diarization | model | custom | oip-v2 / infer | 8403 | CPU |

The two diarization services use the same OIP signature, so apps can switch between them without code
changes. The signature takes a batch of recordings as `AUDIO` (zero-padded rows), `LENGTHS` and
`SAMPLE_RATE`, and it returns `SEGMENTS`, `SPEAKERS` and `ROWS`. `ROWS` gives the recording of each
segment, and in each recording the speakers are numbered in the order they first appear. A request with
one recording as `AUDIO` `[-1]` and no `LENGTHS` still works, and it gets `SEGMENTS` and `SPEAKERS` back
as before.

### Speech enhancement

Speech enhancement services take a recording and return a copy with less noise or less echo from the
room. The copy keeps the sample rate and length of the input. All three services use the same OIP
signature, which takes a batch of recordings as `AUDIO` (zero-padded rows), `LENGTHS` and `SAMPLE_RATE`,
and returns them in the same form. A request with one recording as `AUDIO` `[-1]` and no `LENGTHS` still
works, and it gets one `AUDIO` `[-1]` back.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| deepfilternet3 |  | model | custom | oip-v2 / infer | 8408 | CPU |
| nemotronlabs-re-use |  | model | custom | oip-v2 / infer | 8411 | 4 GB |
| wpe-dereverb |  | utility | custom | oip-v2 / infer | 8502 | CPU |

### Text normalisation

Text normalisation services take ASR text and return a cleaned or reformatted copy, and both use the
`openai-inspired` contract with the `text.transform` API.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| monkey-custom |  | model | custom | openai-inspired / text.transform | 8412 | CPU |
| text-normalizer | default-normalizer | utility | custom | openai-inspired / text.transform | 8501 | CPU |

### Other audio analysis

The remaining audio services analyse recordings in other ways, e.g., speaker verification, speech
quality scores and forced alignment. Forced alignment finds the start and end time of each word in a
known transcript.

| Service | Alias | Kind | Engine | Contract / API | Port | GPU |
|---|---|---|---|---|---|---|
| wavlm-sv |  | model | custom | oip-v2 / infer | 8404 | 3 GB |
| redimnet2-sv |  | model | custom | oip-v2 / infer | 8405 | 2 GB |
| utmos |  | model | custom | oip-v2 / infer | 8406 | CPU |
| qwen3-forced-aligner |  | model | custom | oip-v2 / infer | 8407 | 4 GB |
| dnsmos |  | model | custom | oip-v2 / infer | 8409 | CPU |

## Service names

Service names use kebab case, which means lowercase words joined by hyphens, so `qwen2.5` and
`qwen3.5` become `qwen2-5` and `qwen3-5`. When an engine model's `model` field points at
`/models/...`, the service uses your existing copy of the weights in `$AI_ASSETS`, which is mounted
read-only.

## Port ranges

Each type of service has its own range of ports, and validation gives a warning, not an error, when a
port is outside its range.

| Port range | Services |
|---|---|
| `81xx` | Audio models |
| `82xx` | Chat models and vision language models |
| `83xx` | Embedding and reranking models |
| `84xx` | Other models, usually on the OIP contract |
| `85xx` | Utilities |

The model catalog listens on port `8700`, outside these ranges. Concord Bench, Synthetics and Concord
Label use ports `8600`, `8610` and `8620`.
