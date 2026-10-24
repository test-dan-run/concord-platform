# meralion-3-3b-asr

MERaLiON-3-3B-ASR (A*STAR I2R) speech-to-text on the GPU: Singapore English, Mandarin, Malay, Tamil and
English-X code-switching. Synthetics uses it as the Singlish fallback ASR in stage 5, after `qwen3-asr-1-7b`.

- Contract: `openai` / `audio.transcriptions`, port 8107.
- `response_format`: `json` (`{"text"}`), `text`, or `verbose_json` (adds `duration`; no segments).
- The model picks the language itself; `language`, `prompt` and `temperature` are ignored.
- Audio longer than 30 s is cut into 30 s pieces, and the texts of the pieces are joined with spaces.
- Batching: requests that arrive within `AISVC_BATCH_WAIT_MS` (10 ms when unset) of each other are transcribed
  together. The pieces of all those requests go through the model in one `generate` call per `AISVC_MAX_BATCH`
  pieces (8 when unset). Every piece counts as 30 s for `AISVC_MAX_BATCH_LEN`, because the Whisper encoder pads
  each piece to 30 s. All pieces share the same prompt and the same input size, so no extra padding is needed,
  and greedy decoding gives each piece the same text as it would get alone. In rare cases a word can differ,
  because batched GPU kernels round bf16 numbers a little differently. A file that cannot be decoded fails only
  its own request.
- Weights: `just fetch-weights meralion-3-3b-asr` (6.65 GB). Licence: MERaLiON-3 Public Licence. Using its
  outputs in a product needs a clear acknowledgement, and the Gemma terms apply.
