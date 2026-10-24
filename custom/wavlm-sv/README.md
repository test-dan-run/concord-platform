# wavlm-sv

WavLM-Large + ECAPA-TDNN speaker embeddings (256-d; UniSpeech's speaker-verification model, the SIM model of
Seed-TTS-eval, via `prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native`, CC BY-SA 3.0) on the GPU. Synthetics uses it in stage 5 and the bake-off to check that a cloned voice matches its
reference and that the speakers in a meeting stay apart. `wavlm-sv` and `redimnet2-sv` have equal standing;
each has its own calibrated threshold.

- Contract: `oip-v2` / `infer`, port 8404.
- Input: `AUDIO` FP32 `[-1, -1]` (clips zero-padded to one length), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`
  (8 kHz or more; resampled to 16 kHz).
- Output: `EMBEDDINGS` FP32 `[-1, 256]`, L2-normalised, so cosine similarity is the dot product.
- Weights: `just fetch-weights wavlm-sv` (1.3 GB; the model code comes with the weights).
- Clips shorter than 0.5 s are repeated to 0.5 s; clips longer than 20 s are cut (attention memory). No request parameters.
- `SV_DEVICE=cpu` runs it on the CPU.
