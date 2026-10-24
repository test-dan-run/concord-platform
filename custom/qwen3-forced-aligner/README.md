# qwen3-forced-aligner

Qwen3-ForcedAligner-0.6B (Apache-2.0) word-level forced alignment on the GPU. Synthetics stage 8 aligns each
utterance window of the dry tracks against its `text_norm`.

- Contract: `oip-v2` / `infer`, port 8407.
- Inputs: `AUDIO` FP32 `[-1]` (mono, up to 180 s, the qwen-asr limit), `SAMPLE_RATE` INT32 `[1]` (8 kHz or more), `TEXT` BYTES `[1]`,
  `LANGUAGE` BYTES `[1]` (`en`, `zh`, `yue`, `fr`, `de`, `it`, `ko`, `pt`, `ru`, `es` or their names; no Japanese).
- Outputs: `WORDS` BYTES `[-1]` (the aligner's tokens: punctuation dropped, apostrophes kept, Chinese characters
  one by one), `TIMES` FP32 `[-1, 2]` (start and end in seconds, non-decreasing), `SCORES` FP32 `[-1]` (0 to 1).
- Timestamps come in 80 ms steps. A transcript with no words returns empty outputs.
- Weights: `just fetch-weights qwen3-forced-aligner` (1.84 GB).
