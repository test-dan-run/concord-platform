# aisvc: shared server package for custom services

Installed in both base images (and editable in the platform venv for tests). One app factory per
contract; each adds `GET /health` (503 while `load` runs in the background, 200 when ready),
permissive CORS, and contract-shaped errors. No auth.

```python
from aisvc import create_openai_app, create_openai_inspired_app, create_oip_app

app = create_openai_app("kokoro-tts", "audio.speech", handler, load=model.load)
app = create_openai_inspired_app("text-normalizer", "text.transform", handler, options_model=Options)
app = create_openai_inspired_app("itn-en", "text.itn", handler, options_model=ItnOptionsSubclass)
app = create_oip_app("silero-vad", signature, handler, version="1", load=vad.load)
```

Handler contracts are in each module's docstring (`openai_app.py`, `inspired_app.py`,
`oip_app.py`). Raise `ValueError` for bad input (400); other exceptions become 500.
`aisvc.gpu.limit_torch_memory(torch)` caps PyTorch at the manifest's `gpu.vram_gb`.
