# Why utmos is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** UTMOS22 (strong learner) is a wav2vec 2.0 base encoder with a BLSTM and a frame-wise
  regression head that predicts a mean opinion score from raw audio. It is not a text-generating model, so vLLM
  and SGLang cannot serve it, and speech quality scoring has no OpenAI route.
- **Contract `oip-v2`.** Audio in, scores out. The signature is the batch-shaped one `dnsmos` already uses:
  `AUDIO` FP32 `[-1, -1]` (zero-padded), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]` → `MOS` FP32 `[-1]`.
- **CPU.** About 95 M parameters; scoring a 10 s utterance takes well under a second on 4 cores, and phase C
  already fills the GPU.
- **Weights.** `prj-beatrice/utmos22-torch-native` (MIT), a port of the original checkpoint to plain torch with
  its model code as transformers remote code. Its card checks the outputs against `tarepan/SpeechMOS` v1.2.0.

**Exit condition:** retire this server when an OIP-native engine in `engines.yaml` (MLServer custom runtime or
Triton Python backend) can host a torch model with this pre-processing, keeping the same signature.
