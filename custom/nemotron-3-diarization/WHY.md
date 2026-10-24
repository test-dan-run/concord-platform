# Why nemotron-3-diarization is a custom server

**Checked 2026-10-17.**

- It is a NeMo **Sortformer** diarization checkpoint (`.nemo`). vLLM v0.31.0 and SGLang v0.5.21
  do not load NeMo diarization models.
- **Contract `oip-v2`:** audio array in, speaker segments out; no OpenAI route fits.
- NVIDIA's own runtimes (NeMo, NeMo-Speech.cpp, Riva/NIM) are not engines on this platform. NeMo
  in Python is the reference implementation, and runs here at 0.02× real time on CPU.

**Exit condition:** retire when an engine in `engines.yaml` (for example a Triton/NIM entry with
`contract: oip-v2`) serves Sortformer diarization with this signature.
