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

## Batching

From version 0.2.0, `aisvc.batch` runs many inputs through a model in batches, so an image only writes
a function that runs one padded batch. The function takes a list of items and returns one result per
item, in the same order. `MicroBatcher` sorts the items by length so that each batch needs little
padding, splits them into batches within the limits, and returns the results in the order of the
request. A request with more items than the limits allow is split inside the server and is never
rejected. Calls to the batch function are made one at a time, so the function needs no lock.

```python
import numpy as np
from aisvc import MicroBatcher, create_oip_app
from aisvc.batch import pad, unpad

def embed_batch(clips: list[np.ndarray]) -> list[np.ndarray]:
    audio, lengths = pad(clips)                        # [N, longest] float32 and int64 [N]
    mask = np.arange(audio.shape[1]) < lengths[:, None]  # attention mask from the lengths
    return list(model.forward(audio, mask))            # one embedding per clip

embed = MicroBatcher(embed_batch, length=len)          # length: samples at 16 kHz

def handler(inputs, parameters):
    sr = int(inputs["SAMPLE_RATE"][0])
    clips = [prepare(row, sr) for row in unpad(inputs["AUDIO"], inputs["LENGTHS"])]
    return {"EMBEDDINGS": np.stack(embed(clips))}
```

For an OpenAI route that takes one file per HTTP request, `DynamicBatcher` waits a few milliseconds
after the first request arrives and runs every request that came in meanwhile as one call. Its `key`
function decides which requests can share a batch, so requests with a different language or different
options are run apart. An error in a batch is raised to every request in that batch, so decode and
check each file before you call the batcher. The aisvc routes are plain functions that FastAPI runs in
a thread pool, and each request waits in its own thread for its result.

```python
transcribe = DynamicBatcher(model.transcribe_batch, key=lambda r: (r.language, r.prompt, r.temperature),
                            length=lambda r: len(r.audio))
def handler(req):
    return transcribe(Decoded(decode(req.file), req.language, req.prompt, req.temperature))
```

The limits come from environment variables, which the catalog will set from measured values:

| Variable | Meaning | When unset |
|---|---|---|
| `AISVC_MAX_BATCH` | Most items in one batch. | 8, or the image's own default |
| `AISVC_MAX_BATCH_LEN` | Most padded length in one batch, which is the item count times the longest item. The unit is the one the image's `length` function returns. For audio it is milliseconds of audio (use `aisvc.batch.audio_ms`), so the catalog can set it without knowing the model's sample rate. For text it is tokens. A single item longer than the limit still runs on its own. | No cap, or the image's own default |
| `AISVC_BATCH_WAIT_MS` | How long `DynamicBatcher` waits for more requests after the first one. | 10 |
| `AISVC_VRAM_GB` | The memory cap that `aisvc.gpu.limit_torch_memory` sets for PyTorch. | No cap |

An image passes its own defaults with `MicroBatcher(fn, limits=Limits.from_env(max_items=4, max_len=16000 * 120))`,
and the environment variables still take priority.

When a batch runs out of GPU memory, the batcher empties the CUDA cache, splits the batch in half and
tries again. It also lowers the padded length limit to half of the failed batch, and it keeps the
lower limit for the rest of the process, so a limit that was set too high makes requests slower
instead of making them fail. The batcher logs a warning each time it lowers the limit. A single item
that still runs out of memory raises its error. The batcher knows `torch.cuda.OutOfMemoryError` and
`MemoryError`, and it also matches the out of memory messages that CTranslate2 and ONNX Runtime raise,
so images without torch get the same behaviour.

Every app has `GET /platform/memory`, which returns the GPU memory that PyTorch reports. The body has
the peak allocated and peak reserved memory, the memory allocated and reserved now, and the device's
total and free memory, each in bytes and in GiB. It also has the current batch limits of every
batcher in the process, with the configured values and the number of times the limits were lowered.
`POST /platform/memory/reset` returns the same body and then resets the peak, so the catalog can reset
the peak, send a test load, and read the new peak. When torch or CUDA is missing, the body says
`"torch": false` or `"cuda": false` and still lists the batch limits.

`POST /platform/batch-limits` changes the limits of every batcher while the service runs. The body
`{"max_items": 512, "max_len": null}` sets them, and `{"restore": true}` goes back to the values the
service started with. Either way the count of lowered limits starts again from zero. Each limit must
be a whole number of at least 1 (or null); anything else, such as `2.5` or `true`, is a 422 and changes
nothing. The catalog uses it while it measures an image: it lifts the limits so each test batch runs
whole, and it treats a lowered limit as an out of memory point.
