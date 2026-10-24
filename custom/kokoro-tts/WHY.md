# Why kokoro-tts is a custom server

**Checked 2026-10-17.**

- **vLLM v0.31.0** (the pinned `vllm/vllm-openai` tag): the model registry has no Kokoro / StyleTTS2
  architecture, and `/v1/audio/speech` is not served by `vllm serve`.
- **SGLang v0.5.21**: no Kokoro model and no speech route.
- **vLLM-Omni** (separate project and image, latest v0.30.0 / v0.31.0rc1): its supported-models list
  covers TTS models such as Qwen3-TTS, Voxtral TTS, Fish Speech, IndexTTS-2, MOSS-TTS and
  Higgs-Audio, but **not** Kokoro-82M.

Kokoro is an 82M StyleTTS2-style model with an iSTFTNet vocoder and a separate G2P front end
(misaki + spaCy + espeak-ng). It is not an autoregressive LM, so it gains little from a
paged-attention engine anyway.

**Contract:** `openai` / `audio.speech`. The OpenAI route fits exactly, so it stays on the
first contract even though it is custom.

**Exit condition:** retire this server when vLLM-Omni (or the pinned vLLM/SGLang) serves
`hexgrad/Kokoro-82M` on `/v1/audio/speech`. Replace this manifest with an engine manifest
(`engine: vllm-omni`, added to `engines.yaml`) on the same port and alias. Apps using
`client("default-tts")` will not need to change.
