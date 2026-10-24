# Why dnsmos is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** DNSMOS P.835 is a 1 MB convolutional ONNX model that scores raw 16 kHz audio,
  not a transformer LM. vLLM and SGLang cannot load it, and quality scoring has no OpenAI route.
- **Contract `oip-v2`.** Audio in, scores out. Batch-shaped like the planned `utmos` service:
  `AUDIO` FP32 `[-1, -1]` (zero-padded), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]` → `MOS` (OVRL),
  `SIG`, `BAK` FP32 `[-1]`.
- **Why not Triton's ONNX Runtime backend?** Triton could run the graph, but every clip needs the reference
  pre- and post-processing: resampling, repeat-padding to 9.01 s, 1 s hops, averaging and the polynomial
  calibration. That is a few lines here and a Python backend there.

**Exit condition:** retire this server when an OIP-native engine in `engines.yaml` can host the scoring
loop (MLServer custom runtime or Triton Python backend), keeping the same signature.
