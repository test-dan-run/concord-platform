# Custom model (rung 3): a server on a shared base

Use this only when no pinned engine can serve the model. Typical cases: a non-HF checkpoint format,
a non-LM architecture, or a pipeline with Python logic around the network. You write model loading
and a handler; the `aisvc` factory provides the routes, validation, `/health` (503 while loading)
and CORS.

## Start from the closest existing service

| Contract / api | Copy from | Notes |
|---|---|---|
| `openai` / `audio.speech` | `custom/kokoro-tts` | GPU torch, `limit_torch_memory` |
| `openai` / `audio.transcriptions`, CTranslate2 | `custom/faster-whisper-large-v3` | `py-cpu` base, CUDA from pip wheels; batches the 30 s windows of one file |
| `openai` / `audio.transcriptions`, torch, 30 s window | `custom/ark-asr-0-6b` | cuts audio into 30 s chunks, batches the chunks of concurrent requests |
| `oip-v2` / `infer`, one vector per clip | `custom/redimnet2-sv` | batch signature, `MicroBatcher` over rows, cuts clips at 20 s |
| `oip-v2` / `infer`, CPU | `custom/silero-vad` | `weights.yaml` + sha256, ONNX |
| `oip-v2` / `infer`, GPU | `custom/pyannote-diarization` | variant base `py-cuda-torch211` |
| `oip-v2` / `infer`, existing local weights | `custom/nemotron-3-diarization` | `weights.path` points at an existing dir |
| `oip-v2` / `infer`, speech enhancement with weights | `custom/deepfilternet3` | CPU, `weights.yaml`, AUDIO in and out |
| `oip-v2` / `infer`, speech enhancement, no weights | `custom/wpe-dereverb` | utility, `tasks: [dereverberation, ...]` |
| `openai-inspired` / `text.transform` | `custom/text-normalizer` | utility, no weights |
| `openai-inspired` / `text.itn` | `toy_itn_app` in `tests/factories.py` | no service yet; the toy passes all seven `text.itn` checks |

Read that service's `server/`, `tests/`, `Dockerfile`, `requirements.txt`, `WHY.md` and manifest
before writing anything.

## Files in `custom/<name>/`

- **`Dockerfile`**:

  ```dockerfile
  # <name>: <contract> <what> on <CPU|GPU>. Weights are mounted at /models, never baked in.
  ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.2.0
  FROM ${BASE_IMAGE}
  COPY requirements.txt /app/requirements.txt
  RUN pip install -r /app/requirements.txt
  COPY server/ /app/server/
  ENV WEIGHTS_DIR=/models
  EXPOSE 8000
  CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
  ```

- **`requirements.txt`**: exact pins, with a header comment naming the base and the pin date. Never a
  pip package that bundles weights into the image.
- **`server/__init__.py`, `server/app.py`, `server/<model>.py`**: `build_app(model=None)` constructs the
  model wrapper (lazy: `load()` does the heavy work) and returns one factory call:

  ```python
  create_openai_app(NAME, "audio.speech", handler, load=model.load)            # handler(req) -> result
  create_oip_app(NAME, SIGNATURE, handler, version="...", platform="...", load=model.load)
      # handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]
  create_openai_inspired_app(NAME, "text.transform", handler, options_model=Opts, load=...)
      # handler(texts: list[str], options) -> list[str]
  create_openai_inspired_app(NAME, "text.itn", handler, options_model=Opts, load=...)
      # Opts subclasses aisvc.ItnOptions (language, return_spans, classes; options.class_list splits classes)
      # handler(texts: list[str], options) -> list[tuple[str, list[dict]]]: (output, spans) per input, where
      #   a span is {input_start, input_end, output_start, output_end, class}, half-open code-point offsets
  ```

  For `text.itn`, always compute the spans: the factory drops them unless the request asked for
  `return_spans`, and it checks every span the handler returns (an invalid one is a 500). Reject an
  unsupported `language` or class with a pydantic `field_validator` on `Opts`: the factory turns it into
  400 `invalid_options` (see `toy_itn_app`). The factory also enforces the input limits (256 inputs,
  10,000 characters each).

  Read `NAME` from `AISVC_NAME` and the weights dir from `WEIGHTS_DIR`, both set by render. GPU
  services call `aisvc.gpu.limit_torch_memory(torch)` in `load()`, which caps torch at `AISVC_VRAM_GB`.
  The service runs alone on the GPU, so `AISVC_VRAM_GB` is the memory measured free at start less 1 GiB
  (at most `util_limit` of the GPU), not the manifest's `gpu.vram_gb`. Bad input raises `OIPError` / `OpenAIError` with a
  400, and the factory formats the response.
- **`weights.yaml`** (when weights are fetched rather than already in `$AI_ASSETS`):

  ```yaml
  # Fetched by `just fetch-weights <name>` into ${AI_ASSETS}/custom-models/<name>/<version>.
  version: v1.0
  source: <page or repo at a pinned revision>
  license: <license>
  files:
    - path: <file>
      url: <immutable URL: a tag or commit, never a branch>
      sha256: <sha256sum of the downloaded file>
      size: <bytes>
  ```

  If `weights.yaml` exists, the manifest must have `weights` (validation enforces this).
- **`WHY.md`**: the date checked, the pinned engine versions checked, why no engine (or OIP server)
  can serve it, and an **exit condition** ("retire when ..."). See `custom/silero-vad/WHY.md`.
- **`README.md`**: what it does, the request/response example, parameters, and limits.
- **`tests/test_<name_underscored>.py`**: these run in `just test` with no Docker, GPU or weights:

  ```python
  from conformance import checks_for, load_server, running, service_meta
  SERVICE_DIR = Path(__file__).resolve().parents[1]
  app_mod = load_server(SERVICE_DIR, "app")
  META = service_meta("<name>")

  @pytest.fixture(scope="module")
  def client():
      with running(app_mod.build_app(FakeModel())) as c:   # FakeModel: same interface, no weights
          yield c

  def test_conformance(client):
      for check in checks_for(META):
          check(client, META)
  ```

  For `oip-v2`, the `oip_metadata_matches_signature` check already compares the server's
  `SIGNATURE` with the manifest signature, so no separate test is needed. Add a real-model test
  marked `@pytest.mark.weights` that skips when the weights are missing.

## Batching

Every GPU server batches through `aisvc.batch` (base 0.2.0 and later). Apps send work in concurrent
batches, and the catalog measures how memory grows with batch size, so a server that runs one item at
a time wastes the GPU and cannot be profiled. Read `custom/_bases/common/aisvc/batch.py` (its docstring
is short) and copy the pattern of the closest service:

- **The model wrapper** has one function that runs one padded batch and returns one result per item, in
  order: `transcribe_batch(pieces)`, `embed_batch(clips)`, `enhance_batch(rows)`. Pad the rows and pass an
  attention mask (`aisvc.batch.pad` gives the lengths for it), so each row's result is the same as when
  it runs alone. If the network has no padding mask, run each row at its own length, as
  `own_lengths` in `ark-asr-0-6b` does. The function needs no lock: the batcher serialises its calls.
- **`MicroBatcher(fn, length=lambda x: audio_ms(x, SR), name="...")`** sorts items by length, runs them
  in chunks within the limits, and on out of memory halves the chunk and retries. The length unit is
  milliseconds of audio (`audio_ms`). A model that pads every item to a fixed window gives that window as
  the length of every item (`meralion-3-3b-asr` uses `CHUNK_MS`), because that is what each item costs.
- **An OIP batch signature** (`AUDIO` [-1, -1] + `LENGTHS`) checks the rows, cuts each to its length
  (`rows()` in `server/rows.py` for the diarizers and enhancers, `aisvc.batch.unpad` otherwise) and passes
  them all to the `MicroBatcher` in one call.
- **A route that takes one file per request** (`audio.transcriptions`) wraps the batcher in a
  `DynamicBatcher`, which joins requests that arrive within `AISVC_BATCH_WAIT_MS` (10 ms) into one call.
  Decode and validate each file in the handler before the call, so a bad file fails only its own request.
  See `custom/ark-asr-0-6b/server/app.py`: each file is cut into chunks, and the chunks of all waiting
  requests go through one `MicroBatcher`.
- **A model that batches inside one item** (faster-whisper batches the 30 s windows of one file) reads
  `Limits.from_env()` itself and sizes its inner batch from it, as `batch_size()` in
  `custom/faster-whisper-large-v3/server/model.py` does.
- **Limits** come from `AISVC_MAX_BATCH` and `AISVC_MAX_BATCH_LEN`, which the catalog sets from the
  batch profile. Don't hard-code a batch size. Change the image's default only with a reason, such as
  `Limits.from_env(max_items=1)` in `dnsmos`. `/platform/memory` and `/platform/batch-limits` come with
  the factory; the catalog uses them to measure the image.

Give the manifest `batch` hints when the defaults would measure the wrong thing. They tell the catalog's
sweep which lengths to send and are not part of the config hash:

- `max_item_s`: the longest item the server takes whole, when it cuts longer ones (30 for a 30 s window
  ASR model, 20 for the speaker embedders). The sweep stops there, and a recommendation for a longer
  item gives a note, because the server splits it into several rows.
- `lengths_s`: the lengths to measure at, when short, typical and longest are wrong (`meralion-3-3b-asr`
  pads every piece to 30 s, so it measures only `[30]`).
- `max_n: 1`: the server batches inside one item, so the sweep sends single items of growing length
  (`faster-whisper-large-v3` with `lengths_s: [30, 120, 300, 600]`).

Comment each hint with the reason, as the existing manifests do. Tests (no GPU, with the fake model):
results come back in input order, rows run in batches within the limits set by monkeypatching
`AISVC_MAX_BATCH` / `AISVC_MAX_BATCH_LEN`, and for a `DynamicBatcher`, concurrent requests share one
batch (`ark-asr-0-6b` tests: `test_chunks_run_in_batches_within_the_limits`,
`test_concurrent_requests_share_one_batch`). Add `test_real_model_batch_matches_one_by_one`, marked
`@pytest.mark.weights`, which checks that a padded batch gives the same results as one item at a time.

## Audio longer than the model's window

Many audio models are trained on a fixed window (30 s for Whisper-style encoders), so the server cuts
longer audio into pieces and joins the results. Do it this way:

- Audio no longer than the window is sent whole.
- Longer audio is cut into window-sized chunks, and **a last chunk shorter than 0.1 s is dropped**
  (`MIN_TAIL_S = 0.1`). It holds no speech, and a very short chunk can crash the feature extractor. This
  happened: Concord Label sent `ark-asr-0-6b` a cut of 30 s and 16 samples, the last chunk had 16
  samples, and the mel STFT failed with "Padding size should be less than the corresponding input
  dimension". The request returned a 500 and the pseudo-label run failed. A model that pads every chunk
  to the window would not crash, but it would spend a whole window on the tail, so it gets the same
  rule. Copy `chunks()` from `custom/ark-asr-0-6b/server/ark.py`.
- A whole file shorter than 0.1 s is not run either: it gets empty text, and the model is never
  called. `ark-asr-0-6b` crashed the same way on a file of 200 samples or fewer, and `chunks()` returning
  no pieces handles both cases.
- Set `batch.max_item_s` to the window, so each chunk is one row of a batch to the catalog.

Test the edges with the fake model: 30 s plus 16 samples gives one chunk, 30.1 s gives two, a file of 16
samples gives empty text without calling the model, and a long file is cut into chunks joined in order
(`test_a_tail_under_0_1_s_is_dropped`, `test_a_file_under_0_1_s_gives_empty_text_without_the_model` and
`test_long_audio_is_cut_into_30_s_chunks` in `ark-asr-0-6b`). Real clips are often cut at a round
length plus a few samples, so test just over each window boundary and not only round lengths.

## Manifest fields for a custom model

```yaml
name: <name>
kind: model
engine: custom
contract: oip-v2                    # per the selection table
api: infer
contract_reason: "No OpenAI route for <task>; input is <non-text input>"   # any non-openai contract
port: 84NN
gpu:
  vram_gb: N                        # measured peak, rounded up; omit the gpu block for CPU-only
batch:                              # GPU only, when the defaults measure the wrong thing (see Batching)
  max_item_s: 30                    # longer audio is cut into 30 s pieces that run as separate batch rows
description: "..."
tasks: [<task>, ...]                # kebab-case, from the model card (voice-activity-detection, ...)
build:
  context: custom/<name>
  image: "{registry}/<name>:0.1.0"
  base: py-cpu                      # key in custom/_bases/bases.yaml: py-cpu | py-cuda | py-cuda-torch211 | none
                                    # a new service takes the current version; build.base_version pins an older one
weights:
  path: ${AI_ASSETS}/custom-models/<name>/<version>   # or an existing dir under ${AI_ASSETS}
  mount: /models
custom_reason: >-
  Checked <YYYY-MM-DD>: <why vLLM <tag> and SGLang <tag> cannot serve it>. Retire when <exit condition>.
signature:                          # oip-v2 only
  inputs:
    - {name: AUDIO, datatype: FP32, shape: [-1]}
    - {name: SAMPLE_RATE, datatype: INT32, shape: [1], example: [16000]}   # example: what conformance sends
  outputs:
    - {name: SEGMENTS, datatype: FP32, shape: [-1, 2]}
```

- Give scalar or constrained inputs an `example`. Without one, conformance sends zeros, which a
  `SAMPLE_RATE` check would reject.
- Reuse an existing signature when the task matches. The two diarization services share theirs, so
  apps can switch between them. Speech enhancement uses the batch signature of `deepfilternet3`,
  `nemotronlabs-re-use` and `wpe-dereverb`: `AUDIO` [-1, -1] + `LENGTHS` + `SAMPLE_RATE` in, and the
  same three out, with each row at the rate and length of its input (the catalog fails a length more
  than 10 ms off). The output signature is what makes the catalog verify it as an enhancer. Give a
  batch `AUDIO` input a nested `example` (one short row) and a `LENGTHS` example, as those services
  do, so conformance sends a short clip. Without them, conformance sends one row of 16000 zeros and
  sets `LENGTHS` from it.
- `gpu` takes `vram_gb` only (never `util`). It is the least memory the service needs to load (the
  measured peak, rounded up), and at run time the service gets the whole GPU.

## Verify

```bash
just test                 # in-process conformance with the fake model
just validate && just vram <name> && just render
just fetch-weights <name> # only if the user agreed to download
just build <name>         # builds the base image too if it is missing
just up <name> && just conform <name>
just batch-profile <name> --wait  # GPU only, after catalog registration; leases the model
just batch <name>                 # the limits the next start gets
```

Then go back to step 6 of SKILL.md (pinned tests, README) and report.
