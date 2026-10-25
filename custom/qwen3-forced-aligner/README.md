# qwen3-forced-aligner

Qwen3-ForcedAligner-0.6B (Apache-2.0) word-level forced alignment on the GPU. Synthetics stage 8 aligns each
utterance window of the dry tracks against its `text_norm`.

- Contract: `oip-v2` / `infer`, port 8407.
- Inputs, a batch of clips: `AUDIO` FP32 `[-1, -1]` (one mono clip per row, padded to the longest, each up to 180 s,
  the qwen-asr limit), `LENGTHS` INT32 `[-1]` (the samples of each row), `SAMPLE_RATE` INT32 `[1]` (8 kHz or more, the
  same for every row), `TEXT` BYTES `[-1]` and `LANGUAGE` BYTES `[-1]` (`en`, `zh`, `yue`, `fr`, `de`, `it`, `ko`,
  `pt`, `ru`, `es` or their names; no Japanese). `TEXT` and `LANGUAGE` have one entry per row, or one entry that
  every row uses.
- Outputs: `WORDS` BYTES `[-1]` (the aligner's tokens: punctuation dropped, apostrophes kept, Chinese characters
  one by one), `TIMES` FP32 `[-1, 2]` (start and end in seconds within the row's clip, non-decreasing within a row),
  `SCORES` FP32 `[-1]` (0 to 1), and `ROWS` INT32 `[-1]`, the input row that each word belongs to. The words of
  row 0 come first, then the words of row 1, and so on. A row whose transcript has no words has no entries.
- Timestamps come in 80 ms steps.
- Batching: the server sorts the rows by length and aligns them in padded batches of at most `AISVC_MAX_BATCH`
  rows (8 when unset) and `AISVC_MAX_BATCH_LEN` milliseconds of padded audio. Each row is prepared on its own, the
  audio encoder runs each clip at its own length, and the text positions come from the attention mask, so a row
  gets the same words and the same times as it would alone. Times can differ by one 80 ms step in rare cases,
  because batched GPU kernels round a little differently.
- The single-clip request of 0.1.0 still works. `AUDIO` FP32 `[-1]` without `LENGTHS`, with `TEXT` and `LANGUAGE`
  of shape `[1]`, runs as one row and gets back `WORDS`, `TIMES` and `SCORES` as before, without `ROWS` unless the
  request asks for it. The aicore client sends this request unchanged even though the declared signature is the
  batch one.
- Weights: `just fetch-weights qwen3-forced-aligner` (1.84 GB).
