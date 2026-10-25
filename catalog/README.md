# Model catalog

The catalog is the platform's control plane: one service per compute host that registers, verifies,
schedules and serves every model service. Apps (Concord Bench, Concord Label, Synthetics) never start
a model, verify a model or call a model's port directly. They ask the catalog for models, take
leases for their jobs, and send inference through the catalog's gateway. The design is in
[../MODEL_CATALOG_REDESIGN.md](../MODEL_CATALOG_REDESIGN.md); this file is the contract apps build
against.

- **Address:** `http://127.0.0.1:8700` from the host, `http://catalog:8700` from containers on
  `ai-net`. Apps read it from `AICORE_CATALOG` (set the same value in `CATALOG_URL` if an app has its
  own setting).
- **Run it:** `just catalog-up` (container, restarts with Docker) or `just catalog-serve` (host
  process, for development). `just catalog` still opens the static catalog page.
- **State:** SQLite at `platform/run/catalog/catalog.db`. Model definitions stay in
  `services/*.yaml`; the catalog adds only registration (display name, tags, "derived from"),
  verification results, measured timings, leases, usage, the output index, hosted endpoints and
  export jobs.
- **Secrets:** hosted endpoints' API keys are encrypted (Fernet) under a master key at
  `~/.config/aiplatform/catalog.key` (mode 0600, outside `PLATFORM_DIR`), created by `just catalog-up`
  and mounted as a Compose secret; `CATALOG_MASTER_KEY_FILE` points elsewhere. Without it the catalog
  runs, but adding a hosted model and calling one return 503.
- **No auth**, like the rest of the platform. Ports bind to `127.0.0.1`.

Paths below are under `/api` unless they start with `/m/` or `/v2/` (the gateway).

## Identity: model id and build key

- **Model id** = the platform service name (`whisper-large-v3`). Aliases (`default-asr`) are accepted
  everywhere a model id is, and responses always give the name.
- **Build key** = the exact build that produced an output: 16 hex characters of
  `sha256(image id, config hash, weights hash)`.
  - *image id*: the local Docker image id of the running container (`docker inspect .Image`), or of
    the manifest's image when the model is stopped. A rebuilt image under the same tag is a new build.
  - *config hash*: the manifest fields that change outputs: engine, engine command template, model,
    args, engine env, contract, api, signature, build context and base, weights path. Not port, alias,
    description, tasks or GPU size. Rendered into each container as the label `aiplatform.config_hash`,
    so a running container reports the config it was started with.
  - *weights hash*: custom services with `weights.yaml` hash its file list and sha256s. Every other
    weights folder (engine models' `/models/<dir>`, custom "existing local copy" folders) is hashed by
    content, once, and cached by a quick fingerprint (relative path, size, mtime) so the full hash is
    recomputed only when a file changes. `SOURCE.txt` and hidden files are ignored.
- A build key is `null` while its weights are still being hashed (first run) or its image is not
  present locally. Outputs must not be indexed without one.

## Model record

`GET /api/models` and `GET /api/models/{id}` return:

```json
{
  "id": "whisper-large-v3",
  "aliases": ["default-asr"],
  "registered": true,
  "display_name": "Whisper large-v3",
  "tags": ["call-centre"],
  "parent_id": null,
  "description": "Speech-to-text, Whisper large-v3 (multilingual)",
  "tasks": ["speech-recognition", "speech-translation"],
  "kind": "model", "engine": "vllm", "contract": "openai", "api": "audio.transcriptions",
  "signature": null,
  "gpu": {"fraction": 0.15, "gb": 4.8},
  "state": "running",
  "state_detail": null,
  "leased_by": ["bench:job-12"],
  "in_flight": 0,
  "build_key": "3f9c0a1b2c4d5e6f",
  "image": "vllm/vllm-openai:v0.31.0",
  "capabilities": {
    "build_key": "3f9c0a1b2c4d5e6f",
    "status": "verified",
    "usable_as": "asr",
    "tasks": ["asr"],
    "timing_level": "word",
    "output_detail": ["word_timestamps", "segment_timestamps"],
    "protocol": {"contract": "openai", "api": "audio.transcriptions", "routes": ["/v1/audio/transcriptions"],
                 "response_format": "verbose_json"},
    "rtf": 0.031,
    "verified_at": "2026-10-22T14:03:11Z",
    "report": {"lines": [{"name": "contract", "status": "pass", "reason": "..."}], "sample": {"wer": 0.04}}
  },
  "cold_start_s": 74.2,
  "cold_start_measured": true,
  "endpoint": {"gateway_url": "http://127.0.0.1:8700/m/whisper-large-v3", "base_path": "/v1"},
  "last_used": "2026-10-22T14:10:02Z",
  "calls": 4123,
  "served_name": "whisper-large-v3",
  "hosted": null
}
```

- `state`: `running` | `stopped` | `starting` | `stopping` | `queued` (a request for it is waiting for
  compute) | `failed` (the last start failed; `state_detail` says why). CPU-only models have
  `gpu: null` and are never stopped by the scheduler.
- `capabilities` is the record for the **current** build, or `null` before the first verification.
  `capabilities.status`: `pending` (queued) | `running` | `verified` | `failed` (the report says why) |
  `stale` (verified on an earlier build; the current build has not been verified yet).
- `usable_as` (the probe that applies): `asr` (openai `audio.transcriptions`), `chat-audio` (openai
  `chat.completions` that accepts an `input_audio` part), `segmenter` (oip-v2 with AUDIO and
  SAMPLE_RATE inputs and a SEGMENTS output), `enhancer` (oip-v2 with AUDIO and SAMPLE_RATE in and
  AUDIO out), `itn` (openai-inspired `text.itn`), or `null` (protocol checks only).
- `tasks` inside `capabilities` are what the verifier observed: `asr`, `diar` (segments with
  speakers), `vad` (speech-only segments), `enhance` (speech enhancement), `itn`. KWS joins when a KWS
  model exists. The top-level `tasks` are the manifest's declared tasks.
- `served_name` is what goes in request bodies (`model`): the service name, or a hosted model's
  provider model name. `aicore.model_name()` returns it.
- `timing_level`: `word` | `segment` | `text` for ASR; `segment` for segmenters.
- `cold_start_s` is the median of measured loads for this build (or the model's earlier builds), else
  an estimate (`cold_start_measured: false`): 180 s for engine models, 30 s for custom GPU models,
  10 s for CPU models.

Query parameters for `GET /api/models`: `task` (matches capability tasks or declared tasks; repeat to
OR), `tag` (catalog tags; repeat to AND), `state`, `all=1` (also unregistered platform services; by
default only registered models are listed).

## Registration and verification

| Call | Body | Does |
|---|---|---|
| `POST /api/models/{id}/register` | `{"display_name"?, "tags"?: [], "parent_id"?}` | Registers a platform service (idempotent). Queues verification if the current build has none. 201 with the record. |
| `PATCH /api/models/{id}` | same fields | Edits registration fields. |
| `DELETE /api/models/{id}/register` | | Unregisters (records and index entries are kept). |
| `POST /api/models/{id}/verify` | | Queues verification on demand. 202 with `capabilities` (`pending`). |

Verification takes a lease like any app job (holder `catalog:verify`), so it waits its turn in the
queue and may start the model. It runs the platform's contract conformance checks, then the
capability probe on the 10.56 s two-speaker check clip (`catalog/fixtures/check_clip.wav`): for ASR
it tries word, then segment, then text timing and scores WER; for chat-audio it sends an
`input_audio` part; for segmenters it checks segment bounds and tells VAD from diarisation by the
SPEAKERS output; for enhancers it sends the clip mixed with babble at 5 dB SNR (a synthetic room
response for services whose tasks include `dereverberation`) as JSON tensors and fails a wrong
SAMPLE_RATE, a length more than 10 ms off, NaN values or clipping, then passes when SI-SDR against the
clean clip improves (warns when it does not; both values are in `report.sample`); for ITN it sends
spoken-form phrases with `return_spans` and compares only the digits of each output ("twenty one" →
21, "three thirty p m" → 3:30): all match is a pass, some a warning, none a fail. A model whose footprint
exceeds the whole host is rejected at registration (422). New builds of registered models are verified
automatically.

## Model cards

`GET /api/models/{id}/card` describes the model behind a service, assembled at request time from what
the platform already holds (`catalog/cards.py`); the UI shows it on a model's page under **Model card**
(`/models/{id}?tab=card`).

- `source`: Hugging Face or GitHub repo and revision, from `SOURCE.txt` in the weights folder, the
  manifest comment `# Local copy of <repo> @<sha>`, or the `source` URL in `weights.yaml`; `pinned` is
  false when no revision is recorded. Hosted endpoints show their provider instead.
- `license`: name and link (from `weights.yaml` or the upstream card's header) and the LICENSE text.
- `upstream`: base model, languages and training datasets from the upstream card's YAML header.
  `facts`: architecture, context length, precision, quantization, parameters (from safetensors headers,
  left out for packed 4-bit formats) and size on disk.
- `sections`: the upstream model card cut down to what describes the model: overview, architecture,
  inputs and outputs, languages, training data, intended use and limitations (`markdown`), with its
  evaluation results apart (`evaluation`, shown collapsed) and a link to the full card (`full_url`).
  Sections on installing, running, serving or fine-tuning the model, citations, news and contacts are
  left out, and so are code blocks, badges and raw HTML (HTML tables become Markdown tables). Sections
  are chosen by heading (`cards.digest`). A rule-based service with no model weights has its README,
  without code, in place of an upstream card. `gaps` lists what is missing and how to fill it.

Upstream cards come from the weights folder when it holds one, else from `just fetch-cards [NAME ...]`,
which downloads only README and LICENSE at the recorded revision into `${AI_ASSETS}/model-cards/`
(never into a weights folder: every file there is part of the build key).

## Hosted endpoints

A model a provider already serves online, registered with its contract, base URL, provider model name
and API key. Apps call it through the gateway like any platform model, and never see the key.

| Call | Body | Does |
|---|---|---|
| `POST /api/hosted/test` | the fields below (`key` optional with `id` of an existing one) | Connection test before saving: TLS, key accepted, speaks the contract (`GET /models` lists the model, else one minimal request; OIP: ready + metadata). Stores nothing. |
| `POST /api/hosted` | `{"id", "contract", "api", "base_url", "model", "auth_type": "bearer"\|"header"\|"none", "header_name"?, "key"?, "timeout_s"?: 120, "max_concurrency"?: 4, "max_upload_mb"?: 25, "display_name"?, "tags"?, "parent_id"?, "data_leaves_network": true}` | Creates it and its registration, queues verification. 422 without `data_leaves_network`; 409 if `id` is a platform name or alias. OIP endpoints store the signature from the provider's metadata. |
| `PATCH /api/hosted/{id}` | any of the same | Changes it. A new `key` is tested before it replaces the old one. |
| `DELETE /api/hosted/{id}` | | Removes it, its encrypted key and its registration (`DELETE /api/models/{id}/register` does the same). |
| `POST /api/models/{id}/test` | | The connection test with the stored key. |

- Contracts: openai `audio.transcriptions`, `chat.completions`, `audio.speech`, `embeddings`;
  openai-inspired `text.itn`; oip-v2 `infer`. OpenAI-style base URLs include the version path
  (`https://api.example.com/v1`).
- Record: `state: "hosted"`, `gpu: null`, `image: null`, `cold_start_s: 0`, `served_name` = the provider
  model name, and `hosted: {base_url, model, host, auth_type, header_name, key_hint, timeout_s,
  max_concurrency, max_upload_mb, resolved_url, data_ack_at, next_check_at, last_test}`. The key is
  never returned, logged or exported; `key_hint` is its last four characters.
- Build key: sha256 of contract, api, base URL and model name. Changing any of them is a new build
  (verified again); limits and key replacement are not. Hosted models are also re-verified when their
  last verification is over 24 hours old. Verification takes no lease.
- Leases may name hosted ids next to platform models: they are granted at once and never queue.
  `start` and `stop` return 409.
- Gateway: incoming `Authorization` (and a header named like the custom auth header) is dropped and
  the stored key added; bodies over `max_upload_mb` get 413 before anything is sent; calls past
  `max_concurrency` wait up to `timeout_s`, then 503 `concurrency_limit`; a missing master key gives
  503 `key_unavailable`. For OIP, `/v2/models/{id}/…` is rewritten to the provider model name.

## Export and import

`GET /api/export/models` lists every platform model with its image, image size, weights size and
whether it can be exported (`reason` when not: no local image, no build key yet). Hosted models are
not exported.

| Call | Body | Does |
|---|---|---|
| `POST /api/exports` | `{"models": [...], "destination": "bundle"\|"registry", "registry"?, "images"?, "weights"?, "verification"?}` | Starts the job (202). One at a time: 409 while one runs. |
| `GET /api/exports`, `GET /api/exports/{id}` | | Jobs: `state` queued \| running \| done \| failed \| cancelled, `phase`, `phase_index` / `phase_count`, `bytes_done` / `bytes_expected`, `error`, `failed_phase`, `available`. |
| `DELETE /api/exports/{id}` | | Cancels a running job, or deletes a finished bundle. |
| `GET /api/exports/{id}/download` | | The bundle folder streamed as one `.tar`. |

The bundle is a folder under `CATALOG_EXPORT_DIR` (default `run/catalog/exports`; leave it out of
`dev/.backups` tarballs): `manifest.json`, one `images.tar` from a single `docker save` (each shared
layer once), and `models/<id>/` with `service.yaml`, `registration.json`, `weights/` and
`verification.json`. The job checks free space against the estimate plus 10% first; a failure records
its phase and removes the partial bundle; a restart marks a running job failed. For a registry, each
image is tagged and pushed (the host's `~/.docker/config.json` is mounted read-only for the login)
and `manifest.json` records the pushed refs and digests.

On the target: `python -m catalog import <bundle folder or .tar>` loads the images (`docker load`, or
pulls the pushed refs), writes each `service.yaml` into `services/` (a differing manifest is kept
unless `--force`; nothing is kept if the platform no longer loads), copies weights to where the
manifest mounts them, and registers each model with the running catalog, which verifies every build.
A build key that matches keeps its indexed outputs.

## Leases and the queue

An app takes a lease on the models a job needs, for the length of the job. A leased model is never
stopped. Requests that cannot start yet wait in one strict first-come-first-served queue; no app has
priority.

| Call | Body / response |
|---|---|
| `POST /api/leases` | `{"holder": "bench:job-12", "models": ["pyannote-diarization"], "ttl_s"?: 90}` -> 201 lease |
| `GET /api/leases/{id}` | lease |
| `POST /api/leases/{id}/heartbeat` | lease (409 if it has expired or been released) |
| `DELETE /api/leases/{id}` | release; 204 |
| `GET /api/leases` | live leases and queued requests, queue order |

```json
{"id": "l-8c1f…", "holder": "bench:job-12", "models": ["pyannote-diarization"],
 "state": "queued", "position": 2, "waiting_for": "pyannote-diarization does not fit next to qwen3-32b-awq (leased by label:job-3)",
 "error": null, "created_at": "…", "granted_at": null, "ready_at": null, "expires_at": "…"}
```

- `state`: `queued` -> `starting` (fits; its models are being started) -> `active` (every model is
  healthy and has a build key: call it now, and index what it returns) -> `released` | `expired` | `failed` (a model failed to start, or the request
  can never be met, `error` says why).
- **Heartbeat** every 30 s; a lease (queued or not) expires after `ttl_s` (default 90 s, i.e. three
  missed beats) without one, so a crashed app frees its models and its queue place.
- One request names its whole model set. A request that could only fit by stopping models the same
  holder already leases fails at once (it would wait forever): release the old lease first.
- `aicore.catalog.Catalog.lease(...)` does all of this (heartbeat thread, waiting, release).

### Scheduler rules

- Fit uses declared footprints: `gpu.util` for engines, `gpu.vram_gb / gpu_total_gb` for custom
  services. Free = `util_limit` minus running GPU models minus models of starting/active leases.
- To make room it stops only unleased models with no calls in flight that have run for their settle
  time (2 x cold start, at least 60 s) or whose lease has ended, cheapest *reload cost* first (cold
  start x likelihood of reuse: queued demand and recent use), and only as many as needed.
- One model starts at a time (vLLM sizes its KV cache from free memory at start).
- **Each model loads entirely on the GPU.** Declared footprints leave out other processes and CUDA
  contexts, so the catalog also reads free memory from `nvidia-smi` (the container gets the NVIDIA
  `utility` capability for this). A GPU model starts only when its whole budget (`util` x the device, or
  `vram_gb`) plus 0.5 GiB is free on the device, and never while another GPU model is still loading: vLLM
  assumes the memory around it does not change while it profiles, so a neighbour that loads at the same
  time breaks its start ("No available memory for the cache blocks"). If the measured memory is short,
  the catalog stops more idle models. If none can be stopped, the request waits with
  `waiting_for` saying how much memory is free and about how much other processes use. Just before the
  start it checks again, waiting up to 30 s for stopped models to give their memory back. If memory is
  still short, the request goes back to its place in the queue, and the model is not started at all. A
  model is never left partly loaded. Without `nvidia-smi` the catalog fits by declared footprints only.
- **Models a lease brought up are spun down when nothing needs them.** Once its last lease ends, a model
  is stopped after `CATALOG_RELEASE_LINGER_S` (default 30 s, so a back-to-back lease on the same model
  keeps it loaded) unless a live or queued request names it or a call is in flight. Models started on
  purpose (`POST /api/models/{id}/start`, `just up`, or found running outside the catalog) stay up
  until they are evicted or stopped. A model the gateway started for a call without a lease is stopped
  after `CATALOG_GATEWAY_IDLE_S` (default 600 s) without a call, unless a request names it. A model
  whose lease ended needs no settle time before it is evicted.
- The catalog watches Docker every few seconds, so services started or stopped outside it (`docker`,
  an old `stack.py`) are picked up.

## Gateway

Inference goes through the catalog, addressed by model id, with the platform contract unchanged:

- `/m/{id}/...` forwards to the service root: `/m/whisper-large-v3/v1/audio/transcriptions`,
  `/m/silero-vad/v2/models/silero-vad/infer`.
- `/v2/models/{id}/...` is also accepted for OIP calls.
- With `AICORE_CATALOG` set, `aicore.client(name)` points at the gateway, so app code does not change.
  A name the local endpoint map does not list (a hosted model) is looked up at `/api/models/{name}`
  (cached 60 s); `model_name()` then returns its provider model name.
- A call to a stopped model asks the scheduler to start it and is held until it is healthy, or until
  2 x its measured cold start (10 min while unmeasured). Then 503 with
  `{"error": {"message": "...starting...", "type": "model_starting", "code": 503}}` and `Retry-After`.
  Prefer taking a lease first: then calls never wait on a start.
- Every platform service is reachable through the gateway, registered or not (registration only
  decides what `GET /api/models` lists by default).
- Open calls count as activity: a model with calls in flight is never stopped. Last-used time and call
  counts feed eviction (`GET /api/usage`).

## Output index

Each app stores its own model outputs. The catalog keeps an index so that an output either app has
already produced is not computed again.

Index key: `build_key` + `audio_sha256` (sha256 of the original file's bytes) + span
(`span_start`, `span_end` in seconds, rounded to 3 decimals; `-1, -1` = the whole file) +
`params_hash` (the app's 16-hex-character hash of its request parameters; it must not include the
image or build key). Both Concord apps compute `params_hash` the same way, so their keys agree.

| Call | Body | Response |
|---|---|---|
| `POST /api/index/lookup` | `{"app": "label", "keys": [{"build_key", "audio_sha256", "span_start", "span_end", "params_hash"}]}` | `{"hits": [null \| {"owner": "bench", "output_id": 812, "owner_url": "http://…"}]}`, one per key, in order; the asking app's own entries are preferred |
| `POST /api/index/record` | `{"app": "bench", "entries": [{…key, "output_id": 812}]}` | `{"recorded": n}` |
| `POST /api/index/forget` | `{"app": "bench", "output_ids": [812]}` | `{"forgotten": n}` |
| `POST /api/apps` | `{"name": "bench", "base_url": "http://concord-bench:8000"}` | registers where other apps read this app's outputs |
| `GET /api/apps` | | registered apps |

**What every app serves** for the others (the catalog never stores outputs):

- `POST {base_url}/api/outputs/batch` with `{"ids": [812, 813]}` ->
  `{"outputs": [{"id": 812, "build_key", "audio_sha256", "span_start", "span_end", "params_hash",
  "params", "output", "latency_s"}, …]}` (missing ids are left out). Spans use the index form
  (`-1, -1` for the whole file), and `output` is the app's normalised output (Concord's adapters' shapes).

The reading app copies what it reads into its own output table (keeping the owner's id as
provenance) and records its copy in the index too. A hit whose owner no longer has the output is
simply computed again.

## Job planning

`POST /api/jobs` plans a multi-model job to keep total cold-start time low. It plans; the app runs
the steps, taking one lease per step.

```json
{"pipeline": "staged",
 "stages": [
   {"name": "seg", "models": ["pyannote-diarization", "nemotron-3-diarization"],
    "keys": {"pyannote-diarization": [{"audio_sha256": "…", "span_start": -1, "span_end": -1, "params_hash": "…"}]}},
   {"name": "asr", "models": ["whisper-large-v3", "qwen3-asr-1-7b"], "after": "seg"}
 ]}
```

Response:

```json
{"pipeline": "staged",
 "steps": [{"stage": "seg", "model": "pyannote-diarization", "load": true, "cold_start_s": 21.0,
            "items": 120, "hits": 120, "missing": 0, "skip": true}, …],
 "loads": 3, "est_load_s": 410.0, "hits": 120, "fits_together": false}
```

- `staged` (default): each stage runs over the whole dataset before the next; each model loads once.
  Models already running go first in a stage, then the rest by cold start.
- `co-resident`: both stages at once (streaming, or audio added mid-run). The slowest-loading stage is
  the outer loop and stays up; the other stage's models are re-spun inside it unless they all fit
  beside it, in which case nothing is re-spun.
- `keys` per model are optional. With keys, the catalog checks the index for the current build, and
  a model with nothing missing is skipped (`skip: true`). Keys for a later stage are usually unknown
  until the earlier stage has run (segments decide the cuts); plan again after it.

## Events

`GET /api/events` is a server-sent event stream: `model` events (`{id, state, build_key,
capabilities_status}`), `lease` events (the lease JSON), `queue` events (`{queue: [{id, holder,
position, waiting_for}]}`) and `export` events (the export job JSON). App backends can relay these or poll `GET /api/models` and
`GET /api/leases/{id}`.

## UI

`http://127.0.0.1:8700/` (AeroAudio design system): Models (`?group=none` turns off grouping by model
type), a model's page at `/models/{id}` (apps link there; hosted models get their own layout), Add model
(`/add?name={id}`, `/add?source=hosted`), Export (`/export?model={id}`), Queue and leases, Output index,
Model contract, Docs. Live through `/api/events`.

## Other calls

- `POST /api/models/{id}/start`, `POST /api/models/{id}/stop` (`?force=1` stops a leased model, the
  owner's override). `tools/stack.py up/down` use these when the catalog is running, so `just up`,
  `just down` and Synthetics' `stack.py` calls keep working.
- `GET /api/usage`: calls, errors and last use per model.
- `GET /api/health`.

## Python client (aicore)

```python
from aicore import client
from aicore.catalog import Catalog

cat = Catalog()                                    # AICORE_CATALOG, else http://127.0.0.1:8700
cat.models(task="asr", tag="call-centre")          # list[dict], model records
with cat.lease("bench:job-12", ["whisper-large-v3"], on_wait=print) as lease:
    c = client("whisper-large-v3")                  # openai.OpenAI pointed at the gateway
    ...
hits = cat.index_lookup("bench", keys)              # list[dict | None]
cat.index_record("bench", entries)
plan = cat.plan(stages=[...])
```

`aicore.catalog.FakeCatalog` is an in-memory stand-in with the same methods for app tests (leases are
granted at once; models and capabilities come from the constructor).
