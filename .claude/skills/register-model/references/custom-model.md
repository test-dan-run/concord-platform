# Custom model (rung 3): a server on a shared base

Use this only when no pinned engine can serve the model. Typical cases: a non-HF checkpoint format,
a non-LM architecture, or a pipeline with Python logic around the network. You write model loading
and a handler; the `aisvc` factory provides the routes, validation, `/health` (503 while loading)
and CORS.

## Start from the closest existing service

| Contract / api | Copy from | Notes |
|---|---|---|
| `openai` / `audio.speech` | `custom/kokoro-tts` | GPU torch, `limit_torch_memory` |
| `openai` / `audio.transcriptions` | `custom/faster-whisper-large-v3` | `py-cpu` base, CUDA from pip wheels |
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
  ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1
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
  services call `aisvc.gpu.limit_torch_memory(torch)` in `load()`, which caps torch at the manifest's
  `gpu.vram_gb` (passed in as `AISVC_VRAM_GB`). Bad input raises `OIPError` / `OpenAIError` with a
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

  @pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
  def test_conformance(client, check):
      check(client, META)
  ```

  For `oip-v2`, also assert that the server `SIGNATURE` equals the manifest signature (see
  silero-vad). Add a real-model test marked `@pytest.mark.weights` that skips when the weights are
  missing.

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
profiles: [<profile>]
description: "..."
build:
  context: custom/<name>
  image: "{registry}/<name>:0.1.0"
  base: py-cpu                      # key in custom/_bases/bases.yaml: py-cpu | py-cuda | py-cuda-torch211 | none
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
  apps can switch between them. Speech enhancement uses `deepfilternet3`'s: `AUDIO` + `SAMPLE_RATE` in,
  `AUDIO` + `SAMPLE_RATE` out, with the same rate and length as the input (the catalog fails a length
  more than 10 ms off). The output signature is what makes the catalog verify it as an enhancer.
- `gpu` takes `vram_gb` only (never `util`). It counts as `vram_gb / 32` in the profile budget.

## Verify

```bash
just test                 # in-process conformance with the fake model
just validate && just vram <profile> && just render
just fetch-weights <name> # only if the user agreed to download
just build <name>         # builds the base image too if it is missing
just up <profile> && just conform <name>
```

Then go back to step 6 of SKILL.md (pinned tests, README) and report.
