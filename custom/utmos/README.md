# utmos

UTMOS22 strong learner on CPU: a predicted naturalness MOS for a clip of speech. Synthetics scores every TTS
utterance with it in stage 5 (MOS floor) and in the bake-off. `dnsmos` stays the reference-cleanup scorer.

- Contract: `oip-v2` / `infer`, port 8406, CPU.
- Input: `AUDIO` FP32 `[-1, -1]` (clips zero-padded to one length), `LENGTHS` INT32 `[-1]`, `SAMPLE_RATE` INT32 `[1]`
  (any rate of 8 kHz or more; the model resamples to 16 kHz).
- Output: `MOS` FP32 `[-1]`, about 1 to 5.
- Weights: `just fetch-weights utmos` (406 MB, MIT). The model code (`modeling_*.py`) comes with the weights.
- No request parameters. The model pads clips shorter than 400 samples (25 ms).

```bash
curl -s localhost:8406/v2/models/utmos/infer -H 'content-type: application/json' -d '{"inputs": [
  {"name": "AUDIO", "datatype": "FP32", "shape": [1, 8], "data": [0, 0.1, -0.1, 0.05, 0, -0.05, 0.1, 0]},
  {"name": "LENGTHS", "datatype": "INT32", "shape": [1], "data": [8]},
  {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}]}'
```

