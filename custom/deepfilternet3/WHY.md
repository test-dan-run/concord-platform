# Why deepfilternet3 is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** DeepFilterNet3 is a small recurrent speech-enhancement network (ERB encoder,
  ERB and deep-filter decoders) with a hand-written STFT / ERB / deep-filtering pipeline around it. It is not
  a transformer LM; vLLM and SGLang cannot load it, and speech enhancement has no OpenAI route.
- **Contract `oip-v2`.** Audio in, audio out: `AUDIO` FP32 `[-1]`, `SAMPLE_RATE` INT32 `[1]` → the same.
- **Why the `deep-filter` binary?** The `deepfilternet` pip package (0.5.6, the last release) needs PyTorch
  and a torchaudio API removed in torchaudio 2.1. The project's own Rust CLI runs the same exported ONNX model
  with tract on CPU, has no Python dependencies, and is pinned by sha256 like any other weights file.
- **Why not Triton's ONNX Runtime backend?** The three ONNX graphs need the libDF feature extraction and
  deep-filter synthesis around them (streaming state, ERB bands, complex filtering); that is what the binary does.

**Exit condition:** retire this server when an OIP-native engine in `engines.yaml` can host the libDF
pipeline (for example an MLServer custom runtime), keeping the same signature.
