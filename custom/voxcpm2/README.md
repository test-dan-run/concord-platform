# voxcpm2

VoxCPM2 (OpenBMB, 2B, Apache-2.0) zero-shot voice cloning on the GPU. Synthetics stage 4 clones every utterance from
one clean NSC reference clip per speaker.

- Contract: `oip-v2` / `infer`, port 8410.
- Inputs: `TEXT` BYTES `[1]` (up to 2000 characters), `REF_AUDIO` FP32 `[-1]` (0.25-30 s, mono, any rate of 8 kHz
  or more), `REF_SAMPLE_RATE` INT32 `[1]`, `REF_TEXT` BYTES `[1]` (the reference's transcript), `SEED` INT64 `[1]`.
- Outputs: `AUDIO` FP32 `[-1]`, `SAMPLE_RATE` INT32 `[1]` (48000).
- Parameters (optional):
  - `mode`: `ultimate` (default; the clip is both the continuation prompt and the isolated reference, which the
    model card says gives the highest similarity), `continuation` (prompt and transcript only) or `reference`
    (isolated reference only; `REF_TEXT` is ignored).
  - `cfg_value` (default 2.0) and `inference_timesteps` (default 10).
- The same inputs and `SEED` give the same audio (torch is seeded per request; requests run one at a time).
- Weights: `just fetch-weights voxcpm2` (4.96 GB).
- torch.compile is on (`VOXCPM_OPTIMIZE=1` in the image): RTF 0.22 on the 5090 once warm, against 1.17 without;
  start-up takes about 80 s longer. Set `VOXCPM_OPTIMIZE=0` to turn it off.
