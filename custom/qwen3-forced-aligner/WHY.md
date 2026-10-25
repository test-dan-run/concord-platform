# Why qwen3-forced-aligner is a custom server

Checked 2026-10-18 against vLLM v0.31.0 and SGLang v0.5.21.

- **The engine does not give word timings.** vLLM v0.31.0 registers `Qwen3ASRForcedAlignerForTokenClassification`
  (checked in the image), but only as a token classifier: a caller would have to build the `<timestamp>` slot
  prompt, read per-token classes back from a pooling route and turn them into times. No OpenAI route returns that,
  and the platform's vLLM services speak only `openai`.
- **Contract `oip-v2`.** Audio and text in, word timings out. Since 0.2.0 a request holds a batch of clips:
  `AUDIO` FP32 `[-1, -1]`, `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`, `TEXT` BYTES `[-1]`, `LANGUAGE` BYTES
  `[-1]` → `WORDS` BYTES `[-1]`, `TIMES` FP32 `[-1, 2]`, `SCORES` FP32 `[-1]`, `ROWS` INT32 `[-1]`. The words of all
  rows come in one list, and `ROWS` says which row each word belongs to, because each row has its own number of
  words. The single-clip request of 0.1.0 (`AUDIO` FP32 `[-1]` without `LENGTHS`) still works, so callers can move
  to batches when they are ready.
- **Scores.** The package returns times only. The server keeps the softmax probability of each timestamp slot's
  class, and a word's score is the mean of its two.

**Exit condition:** retire this server when an engine in `engines.yaml` returns word-level forced alignment for this
model behind an OIP route (or an OpenAI route gains forced alignment).
