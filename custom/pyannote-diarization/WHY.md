# Why pyannote-diarization is a custom server

**Checked 2026-10-17.**

- It is a multi-stage pipeline (PyanNet segmentation → WeSpeaker embeddings → agglomerative
  clustering → reconstruction), not a transformer LM. vLLM and SGLang cannot run it.
- **Contract `oip-v2`:** the input is an audio array. OpenAI has no diarization-only route; its
  `diarized_json` transcription format needs a transcript and is served by ASR models.
- **Why not Triton or MLServer?** An OIP-native server could host the two networks, but the
  clustering and reconstruction steps are Python pipeline logic. That would need a Python backend
  that ends up equivalent to this server.

**Exit condition:** retire when an OIP engine added to `engines.yaml` runs pyannote pipelines
(e.g. an MLServer custom runtime) with the same signature, or when `nemotron-3-diarization`
fully replaces it in practice. The two are interchangeable by signature.
