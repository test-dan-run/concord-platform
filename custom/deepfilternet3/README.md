# deepfilternet3

Speech denoising with DeepFilterNet3 (v0.5.6, `deep-filter` Rust CLI on CPU). Used by Synthetics to clean
NSC reference clips before cloning (denoise here, then `wpe-dereverb`).

- Contract: `oip-v2` / `infer`, port 8408, CPU.
- Input: `AUDIO` FP32 `[-1, -1]` (mono clips padded with zeros to one length), `LENGTHS` INT32 `[-1]`,
  `SAMPLE_RATE` INT32 `[1]` (8 kHz or more, one rate for every clip).
- Output: `AUDIO` FP32 `[-1, -1]` padded with zeros, `LENGTHS` INT32 `[-1]` and `SAMPLE_RATE` INT32 `[1]`. Each row
  keeps its input's rate and length. The signature is the same as `nemotronlabs-re-use` and `wpe-dereverb`.
- One clip: a request with `AUDIO` FP32 `[-1]` and no `LENGTHS` still works, and it gets `AUDIO` FP32 `[-1]` back.
- Batches: the clips run one after another, one `deep-filter` run each. The CLI can take several files in one
  run, but it carries the model state from one file to the next, which changed the output of the later files
  (checked 2026-10-25), so it is not used that way. Clips shorter than 0.1 s are padded with silence before the
  run, because the CLI crashes on very short input.
- Parameters: `atten_lim_db` (0-100, default 100 = full reduction), `post_filter` (default false).
- Weights: `just fetch-weights deepfilternet3` (binary + `DeepFilterNet3_onnx.tar.gz`, MIT OR Apache-2.0).

```python
out = client("deepfilternet3").infer({"AUDIO": audio, "SAMPLE_RATE": np.array([16000], np.int32)})  # one clip
clean = out["AUDIO"]
```

Limits: JSON tensors, so keep requests to clips of a minute or less. The model runs at 48 kHz internally.
