# Why monkey-custom is a custom server

Checked 2026-10-24 against vLLM v0.31.0 and SGLang v0.5.21.

- **No engine can run it.** It is [transcript-normer](https://github.com/test-dan-run/transcript-normer)
  (commit e98d4ae): regular expressions with context guards for fillers, Singlish particles and acronyms,
  whisper-normalizer for canonical ITN, and a 210 MB ONNX punctuation/casing model run with onnxruntime. None
  of it is a generative LM.
- **Contract `openai-inspired` / `text.transform`.** Text in, text out, one output per input. Not `text.itn`:
  it deletes fillers and particles and, in canonical mode, lowercases and drops punctuation, so outputs
  cannot be mapped back to input spans. The project's own `/api/v1/normalize` request fields become scalar
  `options` (`remove_fillers`, `remove_particles`, `mode`).
- **Code.** `server/normer/` is the project's `app/services/` at e98d4ae, changed only to relative imports.
  The project's own FastAPI layer is not used; `aisvc` provides the contract's routes, errors and `/health`.
- **Weights.** The project bakes the punctuation model into its image; here it is fetched once by
  `just fetch-weights monkey-custom` and mounted at `/models`, and the service never downloads at start.

**Exit condition:** retire this server if transcript-normer itself serves `text.transform` (then register
its image directly), or if a platform engine gains a rule-based text pipeline runtime.
