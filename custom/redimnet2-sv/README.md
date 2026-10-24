# redimnet2-sv

ReDimNet2-B6 speaker embeddings (192-d, PalabraAI/redimnet2, VoxBlink2 + VoxCeleb2 + CN-Celeb2, large-margin fine-tuned, MIT) on the GPU. Synthetics uses it in stage 5 and the bake-off to check that a cloned voice matches its
reference and that the speakers in a meeting stay apart. `wavlm-sv` and `redimnet2-sv` have equal standing;
each has its own calibrated threshold.

- Contract: `oip-v2` / `infer`, port 8405.
- Input: `AUDIO` FP32 `[-1, -1]` (clips zero-padded to one length), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`
  (8 kHz or more; resampled to 16 kHz).
- Output: `EMBEDDINGS` FP32 `[-1, 192]`, L2-normalised, so cosine similarity is the dot product.
- Weights: `just fetch-weights redimnet2-sv` (51 MB plus the model code).
- Clips shorter than 0.5 s are repeated to 0.5 s; clips longer than 20 s are cut (activation memory). No request parameters.
- `SV_DEVICE=cpu` runs it on the CPU.
