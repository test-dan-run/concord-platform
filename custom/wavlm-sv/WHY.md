# Why wavlm-sv is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** WavLM-Large with an ECAPA-TDNN head over its layer-weighted features returns a 256-d
  speaker embedding. vLLM's pooling runner covers text embedding models only, SGLang has no WavLM, and the OpenAI
  `embeddings` route takes text, not audio.
- **Contract `oip-v2`.** Audio in, embeddings out: `AUDIO` FP32 `[-1, -1]` (zero-padded), `LENGTHS` INT32 `[-1]`,
  `SAMPLE_RATE` INT32 `[1]` → `EMBEDDINGS` FP32 `[-1, -1]` (L2-normalised). `redimnet2-sv` has the same signature.
- **Why not `microsoft/wavlm-base-plus-sv`.** On NSC clips its x-vector cosines overlapped between speakers
  (2026-10-18: same-speaker min 0.90, different-speaker max 0.95); this model kept them apart (means 0.79 vs 0.21).
- **Licence.** CC BY-SA 3.0 (UniSpeech). Synthetics uses the embeddings only to check its own audio.

**Exit condition:** retire this server when an engine in `engines.yaml` serves audio speaker-embedding models
behind an OIP or embeddings route, keeping this signature.
