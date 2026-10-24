# ark-asr-0-6b

ARK-ASR-0.6B (Edge0, formerly AutoArk-AI) speech-to-text on the GPU: 19 languages (Chinese, English, German,
Japanese, French, Korean, Spanish, Polish, Italian, Romanian, Hungarian, Czech, Dutch, Finnish, Croatian, Slovak,
Slovene, Estonian, Lithuanian). Trained by on-policy distillation from a larger ASR teacher; its card puts it between
Qwen3-ASR-0.6B and Qwen3-ASR-1.7B on English WER and Chinese CER.

- Contract: `openai` / `audio.transcriptions`, port 8109.
- `response_format`: `json` (`{"text"}`), `text`, or `verbose_json` (adds `duration`; no segments).
- The model picks the language itself; `language`, `prompt` and `temperature` are ignored.
- Audio longer than 30 s is cut into 30 s pieces, and the texts of the pieces are joined with spaces (also for
  Chinese and Japanese). Each piece produces at most 256 tokens, as in the authors' script.
- Batching: requests that arrive within `AISVC_BATCH_WAIT_MS` (10 ms when unset) of each other are transcribed
  together. The server sorts the pieces of all those requests by length and runs them through `generate` in
  batches of at most `AISVC_MAX_BATCH` pieces (8 when unset) and `AISVC_MAX_BATCH_LEN` milliseconds of padded
  audio. Each piece is prepared on its own and the prompts are padded on the left. The audio encoder has no
  padding mask, so the server encodes each piece at its own length (pieces of the same length together) instead
  of padding the audio. Greedy decoding then gives each piece the same text as it would get alone. In rare cases
  a word can differ, because batched GPU kernels round bf16 numbers a little differently. A file that cannot be
  decoded fails only its own request.
- Weights: `just fetch-weights ark-asr-0-6b` (2.6 GB). Licence: Apache-2.0.
