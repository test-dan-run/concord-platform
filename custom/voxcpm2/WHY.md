# Why voxcpm2 is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the vLLM-Omni model list.

- **No engine can run it.** VoxCPM2 is a tokenizer-free diffusion-autoregressive TTS model (MiniCPM-4 backbone,
  local diffusion transformer, AudioVAE V2 decoder). vLLM and SGLang have no implementation of it. The authors'
  Nano-vLLM port is a separate fork, not an engine in `engines.yaml`.
- **Contract `oip-v2`, not `audio.speech`.** Cloning needs a reference audio array plus its transcript for each
  request. OpenAI's `audio.speech` takes only a voice name. Pre-registering the 175 NSC pool voices as named voices
  was rejected, because Synthetics swaps references per meeting.
- **Signature.** `TEXT` BYTES `[1]`, `REF_AUDIO` FP32 `[-1]`, `REF_SAMPLE_RATE` INT32 `[1]`, `REF_TEXT` BYTES `[1]`,
  `SEED` INT64 `[1]` → `AUDIO` FP32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`.

**Exit condition:** retire this server when vLLM-Omni (or the pinned vLLM) serves VoxCPM2 with per-request
reference audio and transcript, or when an OpenAI-style speech route gains reference-audio input.
