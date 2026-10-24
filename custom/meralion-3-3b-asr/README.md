# meralion-3-3b-asr

MERaLiON-3-3B-ASR (A*STAR I2R) speech-to-text on the GPU: Singapore English, Mandarin, Malay, Tamil and
English-X code-switching. Synthetics uses it as the Singlish fallback ASR in stage 5, after `qwen3-asr-1-7b`.

- Contract: `openai` / `audio.transcriptions`, port 8107.
- `response_format`: `json` (`{"text"}`), `text`, or `verbose_json` (adds `duration`; no segments).
- The model picks the language itself; `language`, `prompt` and `temperature` are ignored.
- Audio longer than 30 s is cut into 30 s pieces, transcribed one by one and joined with spaces.
- Weights: `just fetch-weights meralion-3-3b-asr` (6.65 GB). Licence: MERaLiON-3 Public Licence. Using its
  outputs in a product needs a clear acknowledgement, and the Gemma terms apply.
