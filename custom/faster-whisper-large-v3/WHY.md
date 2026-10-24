# Why faster-whisper-large-v3 is a custom server

**Checked 2026-10-17.**

- The weights are a **CTranslate2** conversion (`model.bin` + vocabulary). vLLM v0.31.0 and
  SGLang v0.5.21 load Hugging Face transformers checkpoints only.
- The same model in HF format already runs on vLLM as `whisper-large-v3` (rung 1 of the ladder).
  This service exists for **word-level timestamps**: vLLM's `/v1/audio/transcriptions`
  `verbose_json` returns segments only, with no `timestamp_granularities=word`.
- **Contract `openai`:** OpenAI's transcription route covers it exactly, including `verbose_json`
  `words`.

**Exit condition:** retire when the vLLM `whisper-large-v3` service returns word timestamps
(`timestamp_granularities[]=word`). Apps that need words then switch to `default-asr`.
