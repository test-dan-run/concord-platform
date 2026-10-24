# Why redimnet2-sv is a custom server

Checked 2026-10-18 against vLLM v0.31.0, SGLang v0.5.21 and the Triton/MLServer docs.

- **No engine can run it.** ReDimNet2 is a 2D/1D convolutional speaker encoder over log-mel features, not a
  transformer LM. vLLM and SGLang cannot load it, and no OpenAI route takes audio for embeddings.
- **No package.** The authors ship it as a torch.hub repo (PalabraAI/redimnet2, MIT) whose loader downloads at run
  time. Here the code at commit c5bbe0b is fetched with the checkpoint by `just fetch-weights` and imported from
  the weights mount, so the image downloads nothing at run time.
- **Contract `oip-v2`.** Same signature as `wavlm-sv`: `AUDIO` FP32 `[-1, -1]`, `LENGTHS` INT32 `[-1]`,
  `SAMPLE_RATE` INT32 `[1]` → `EMBEDDINGS` FP32 `[-1, -1]` (192-d, L2-normalised).

**Exit condition:** retire this server when an OIP-native engine (Triton with the PyTorch backend, or MLServer)
can host the model with its feature front end, keeping this signature.
