# wpe-dereverb

Single-channel dereverberation with weighted prediction error (WPE, `nara_wpe` 0.0.11). A statistical
method, not a neural model, so it is a utility with no weights. Used by Synthetics after `deepfilternet3`
to make NSC reference clips dry before voice cloning.

- Contract: `oip-v2` / `infer`, port 8502, CPU.
- Input: `AUDIO` FP32 `[-1, -1]` (mono clips padded with zeros to one length), `LENGTHS` INT32 `[-1]`,
  `SAMPLE_RATE` INT32 `[1]` (8 kHz or more, one rate for every clip). The clips run one after another.
- Output: `AUDIO` FP32 `[-1, -1]` padded with zeros, `LENGTHS` INT32 `[-1]` and `SAMPLE_RATE` INT32 `[1]`. Each row
  keeps its input's rate and length. The signature is the same as `nemotronlabs-re-use` and `deepfilternet3`.
- One clip: a request with `AUDIO` FP32 `[-1]` and no `LENGTHS` still works, and it gets `AUDIO` FP32 `[-1]` back.
- Parameters: `taps` (10), `delay` (3), `iterations` (3). STFT of 32 ms with an 8 ms shift.

Clips shorter than four STFT windows are returned unchanged. Keep requests to a minute or less (JSON tensors).
