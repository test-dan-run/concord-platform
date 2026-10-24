# Why ark-asr-0-6b is a custom server

Checked 2026-10-23 against vLLM v0.31.0 and SGLang v0.5.21.

- **Not in the pinned engine.** `ArkasrForConditionalGeneration` (a Whisper-large-style encoder with RoPE in place of
  absolute positions, a 4x frame-merging MLP adapter, and a Qwen2 decoder of 0.5B size) is the authors' own
  transformers remote code. It is not in vLLM v0.31.0's model registry (checked in the image), and the authors
  publish no vLLM or SGLang integration.
- **Rung 3, not a vLLM plugin (user decision, 2026-10-23).** A plugin would mean writing and maintaining the model
  class for vLLM and a new engine image. Instead this serves the model's own remote code with the prompt, greedy
  decoding, token blocking and text cleanup from the authors' script (`server/ark.py`), as `meralion-3-3b-asr` does.
- **Contract `openai` / `audio.transcriptions`.** Audio in, text out: the OpenAI route fits.

**Exit condition:** retire this server when the pinned vLLM supports ARK-ASR (natively or through a plugin that loads
into it), and register it as an engine service with the same name.
