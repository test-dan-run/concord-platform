# ark-asr-0-6b

ARK-ASR-0.6B (Edge0, formerly AutoArk-AI) speech-to-text on the GPU: 19 languages (Chinese, English, German,
Japanese, French, Korean, Spanish, Polish, Italian, Romanian, Hungarian, Czech, Dutch, Finnish, Croatian, Slovak,
Slovene, Estonian, Lithuanian). Trained by on-policy distillation from a larger ASR teacher; its card puts it between
Qwen3-ASR-0.6B and Qwen3-ASR-1.7B on English WER and Chinese CER.

- Contract: `openai` / `audio.transcriptions`, port 8109.
- `response_format`: `json` (`{"text"}`), `text`, or `verbose_json` (adds `duration`; no segments).
- The model picks the language itself; `language`, `prompt` and `temperature` are ignored.
- Audio longer than 30 s is cut into 30 s pieces, transcribed one by one and joined with spaces (also for Chinese
  and Japanese). Each piece produces at most 256 tokens, as in the authors' script.
- Weights: `just fetch-weights ark-asr-0-6b` (2.6 GB). Licence: Apache-2.0.
