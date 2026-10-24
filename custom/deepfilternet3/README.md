# deepfilternet3

Speech denoising with DeepFilterNet3 (v0.5.6, `deep-filter` Rust CLI on CPU). Used by Synthetics to clean
NSC reference clips before cloning (denoise here, then `wpe-dereverb`).

- Contract: `oip-v2` / `infer`, port 8408, CPU.
- Input: `AUDIO` FP32 `[-1]` (mono, any rate ≥ 8 kHz), `SAMPLE_RATE` INT32 `[1]`.
- Output: `AUDIO` FP32 `[-1]` at the input rate and length, `SAMPLE_RATE` INT32 `[1]`.
- Parameters: `atten_lim_db` (0-100, default 100 = full reduction), `post_filter` (default false).
- Weights: `just fetch-weights deepfilternet3` (binary + `DeepFilterNet3_onnx.tar.gz`, MIT OR Apache-2.0).

```python
out = client("deepfilternet3").infer({"AUDIO": audio, "SAMPLE_RATE": np.array([16000], np.int32)})
clean = out["AUDIO"]
```

Limits: JSON tensors, so keep requests to clips of a minute or less. The model runs at 48 kHz internally.
