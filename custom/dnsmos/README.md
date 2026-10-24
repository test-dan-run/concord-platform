# dnsmos

DNSMOS P.835 speech quality (Microsoft DNS Challenge, non-personalised model) on CPU. Synthetics uses the
overall score to rank cleaned NSC reference clips and keep the best few per speaker.

- Contract: `oip-v2` / `infer`, port 8409, CPU.
- Input: `AUDIO` FP32 `[-1, -1]` (clips zero-padded to one length), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`.
- Output: `MOS` FP32 `[-1]` (OVRL, about 1-5), `SIG` (speech distortion), `BAK` (background noise).
- Weights: `just fetch-weights dnsmos` (`sig_bak_ovr.onnx`, CC-BY-4.0).

Clips shorter than 9.01 s are repeated to that length, as the reference script does.
