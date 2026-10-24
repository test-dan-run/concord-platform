# Why nemotronlabs-re-use is a custom server

Checked 2026-10-24 against vLLM v0.31.0, SGLang v0.5.21 and the model repo at revision 31f82f2.

- **No engine can run it.** NVIDIA NemotronLabs RE-USE is a 9.6M-parameter universal speech enhancement model:
  a convolutional encoder and decoder around 30 bidirectional Mamba (selective state space) blocks, applied to
  STFT magnitude and phase. It is not a transformer LM, so vLLM and SGLang cannot load it, and speech
  enhancement has no OpenAI route.
- **Contract `oip-v2`.** Audio in, audio out, the same signature as `deepfilternet3`: `AUDIO` FP32 `[-1]` and
  `SAMPLE_RATE` INT32 `[1]`, returned at the input's rate and length.
- **Why compile mamba-ssm?** The Mamba blocks call mamba-ssm's CUDA selective-scan kernel (the model has no CPU
  mode). mamba-ssm 2.3.2.post1 ships wheels for CUDA 13 only up to torch 2.10, and the platform's GPU base has
  torch 2.14, so the Dockerfile builds the kernel for sm_120 against that torch. The repo's own scripts use
  files, torchaudio and the Hub; `server/enhancer.py` runs the same steps on arrays in memory.
- **Model code.** `server/reuse/` is the repo's `models/` at revision 31f82f2 (OpenMDW-1.1, `server/reuse/LICENSE`),
  changed only to drop the Hub mixin.

**Exit condition:** retire this server when an OIP-native engine in `engines.yaml` can host the model (for
example a TensorRT or ONNX export of RE-USE served by Triton), keeping the same signature.
