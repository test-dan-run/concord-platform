# Why silero-vad is a custom server

Checked 2026-10-17 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** Silero VAD is a 2 MB recurrent ONNX model (not a transformer LM).
  vLLM and SGLang only load HF-style architectures from their registries, and VAD has no OpenAI route.
- **Contract `oip-v2`.** Its input is a float audio array, so by the selection order (`openai` →
  `oip-v2` → `openai-inspired`) it is OIP v2. Signature: `AUDIO` FP32 `[-1]`, `SAMPLE_RATE`
  INT32 `[1]` → `SEGMENTS` FP32 `[-1, 2]` (start/end seconds).
- **Why not Triton's ONNX Runtime backend?** Triton could serve the raw ONNX graph over OIP, but
  each request needs a loop over 32 ms frames that carries the recurrent `state` and 64-sample
  context, then hysteresis/padding post-processing to turn probabilities into segments. That
  would need a Triton Python backend or BLS ensemble, which is more machinery than this ~150-line
  FastAPI service for a CPU model.

**Exit condition:** retire this server when an OIP-native engine is added to `engines.yaml`
(Triton or MLServer) and the frame loop + segment logic run there (for example as an MLServer
custom runtime or a Triton Python-backend model), keeping the same signature.
