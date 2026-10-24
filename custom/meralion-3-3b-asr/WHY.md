# Why meralion-3-3b-asr is a custom server

Checked 2026-10-18 against vLLM v0.31.0 and SGLang v0.5.21.

- **Not in the pinned engine.** `MERaLiON3ASRForConditionalGeneration` (Whisper-large-v3 encoder + Gemma-2 decoder) is
  not in vLLM v0.31.0's model registry (checked in the image). The authors' vLLM plugin (`meralion-3-asr` 0.1.0)
  targets vLLM 0.12-0.16 and pins `vllm==0.16.0`, so it does not load into the pinned v0.31.0. A second vLLM pin
  for one model was rejected (2026-10-18, user decision): it adds a ~20 GB engine image to shared infrastructure.
- **Rung 3 instead.** The model's own transformers remote code, with the package's fixed prompt, greedy decoding
  (6-gram repeat guard) and 30 s chunking copied into `server/meralion.py`. The card calls this backend
  experimental; the Synthetics bake-off measures its Singlish WER directly.
- **Contract `openai` / `audio.transcriptions`.** Audio in, text out: the OpenAI route fits.

**Exit condition:** retire this server when the pinned vLLM supports MERaLiON-3 (natively or through a plugin
that loads into it), and register it as an engine service with the same name.
