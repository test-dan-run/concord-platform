# Why nemotronlabs-re-use is a custom server

Checked 2026-10-24 against vLLM v0.31.0, SGLang v0.5.21 and the model repo at revision 31f82f2.

- **No engine can run it.** NVIDIA NemotronLabs RE-USE is a 9.6M-parameter universal speech enhancement model:
  a convolutional encoder and decoder around 30 bidirectional Mamba (selective state space) blocks, applied to
  STFT magnitude and phase. It is not a transformer LM, so vLLM and SGLang cannot load it, and speech
  enhancement has no OpenAI route.
- **Contract `oip-v2`.** Audio in, audio out, the same signature as `deepfilternet3` and `wpe-dereverb`: a
  batch as `AUDIO` FP32 `[-1, -1]` (zero-padded), `LENGTHS` INT32 `[-1]` and `SAMPLE_RATE` INT32 `[1]`, returned
  in the same form at each input's rate and length. One clip as `AUDIO` `[-1]` without `LENGTHS` still works.
- **Why no padded batches?** Padding would change the result, for three reasons. The InstanceNorm layers take
  their statistics over the whole spectrogram, so padded frames would change them. The Mamba blocks run both
  forwards and backwards in time, so the backward pass would start in the padding. The STFT also reflects the
  signal at its edges. So only pieces of the same length share a forward pass. The chunks of a long file all
  have the same length, so they run together, and so do clips of equal length.
- **Why compile mamba-ssm?** The Mamba blocks call mamba-ssm's CUDA selective-scan kernel (the model has no CPU
  mode). mamba-ssm 2.3.2.post1 ships wheels for CUDA 13 only up to torch 2.10, and the platform's GPU base has
  torch 2.14, so the Dockerfile builds the kernel for sm_120 against that torch. The repo's own scripts use
  files, torchaudio and the Hub; `server/enhancer.py` runs the same steps on arrays in memory.
- **Model code.** `server/reuse/` is the repo's `models/` at revision 31f82f2 (OpenMDW-1.1, `server/reuse/LICENSE`),
  changed only to drop the Hub mixin.

**Exit condition:** retire this server when an OIP-native engine in `engines.yaml` can host the model (for
example a TensorRT or ONNX export of RE-USE served by Triton), keeping the same signature.
