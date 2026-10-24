# wpe-dereverb

Single-channel dereverberation with weighted prediction error (WPE, `nara_wpe` 0.0.11). A statistical
method, not a neural model, so it is a utility with no weights. Used by Synthetics after `deepfilternet3`
to make NSC reference clips dry before voice cloning.

- Contract: `oip-v2` / `infer`, port 8502, CPU.
- Input: `AUDIO` FP32 `[-1]` (mono, ≥ 8 kHz), `SAMPLE_RATE` INT32 `[1]`.
- Output: `AUDIO` FP32 `[-1]` (same rate and length), `SAMPLE_RATE` INT32 `[1]`.
- Parameters: `taps` (10), `delay` (3), `iterations` (3). STFT of 32 ms with an 8 ms shift.

Clips shorter than four STFT windows are returned unchanged. Keep requests to a minute or less (JSON tensors).
