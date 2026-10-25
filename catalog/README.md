# Model Catalog

The catalog is the platform's control plane: one service per compute host that registers, verifies,
schedules and serves every model service. Apps (Concord Bench, Concord Label, Synthetics) never start
a model, verify a model or call a model's port directly. They ask the catalog for models, take
leases for their jobs, and send inference through the catalog's gateway. The design is in
[../MODEL_CATALOG_REDESIGN.md](../MODEL_CATALOG_REDESIGN.md); this file is the contract apps build
against.

The same service also runs the Dataset Catalog, which keeps versioned datasets and their audio for both
Concord apps. It has its own section, [Dataset Catalog](#dataset-catalog), near the end of this file. In
this file, "the catalog" means the catalog service.

- **Address:** `http://127.0.0.1:8700` from the host, `http://catalog:8700` from containers on
  `ai-net`. Apps read it from `AICORE_CATALOG` (set the same value in `CATALOG_URL` if an app has its
  own setting).
- **Run it:** `just catalog-up` (container, restarts with Docker) or `just catalog-serve` (host
  process, for development). `just catalog` still opens the static AI service list.
- **State:** SQLite at `platform/run/catalog/catalog.db`. Model definitions stay in
  `services/*.yaml`; the catalog adds only registration (display name, tags, "derived from"),
  verification results, measured timings, leases, usage, the output index, hosted endpoints and
  export jobs.
  The Dataset Catalog keeps its state apart, in the dataset store (see [The store](#the-store)).
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
  AUDIO out), `itn` (openai-inspired `text.itn`), or `null` (protocol checks only). AUDIO may be one
  clip (`[-1]`) or a batch (`[-1, -1]` with LENGTHS). The probe sends a batch signature a batch of one
  row, and it checks that a diarizer's ROWS output puts every segment in row 0.
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
| `PATCH /api/models/{id}` | same fields | Edits the fields given; `null` clears one (`"parent_id": null` removes "Derived from"). Registering again leaves out a field that is missing or `null`. |
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

## Batch profiles

The catalog measures how a GPU model's peak memory grows with batch size, so it can recommend the largest
batch that fits a given amount of memory (`catalog/batching.py`). A profile is stored per build key,
because a rebuilt image can use memory differently.

| Call | What it does |
|---|---|
| `POST /api/models/{id}/batch-profile` | Queues a measurement. 202 with `measuring` (`pending` or `running`). |
| `GET /api/models/{id}/batch?vram_gb=24&item_s=20&n=8` | The current build's profile and a `table` of item lengths against 16, 24 and 32 GB and the full budget. With `item_s` (or `item_tokens` for vLLM), `recommendation` is the largest batch for `vram_gb` (default: the full budget). With `n` too, `need` is the memory that batch needs. A length in the other unit, or an audio length over `batch.max_item_s` (the server splits it), gives a `note` instead of both. For a custom GPU image, `limits` are the batch limits its next start gets with the full budget, with the `source` of each (`measured`, `override` or `default`), and `override` is what was set by hand. `running` is null when the service is stopped. Otherwise it has the limits the catalog started it with (`started_with`, null when it was started outside the catalog) and the limits each batcher in the service reports now (`batchers`), including any it lowered after running out of memory. |
| `PUT /api/models/{id}/batch-override` | Sets the batch limits by hand: `max_batch` (most inputs in one batch) and `max_len_s` (most audio in one batch, in seconds). A limit left out still comes from the profile. The override is kept per model, so it stays in place after a rebuild, and it applies from the next start. 409 for a vLLM or CPU service. |
| `DELETE /api/models/{id}/batch-override` | Removes the override, so the next start uses the measured limits again. |

For a custom audio image, the measurement takes a lease (holder `catalog:batch`), so it waits its turn.
It lifts the server's batch limits (`POST /platform/batch-limits` in `aisvc`), resets the peak, and sends
batches of 1, 2, 4 and so on up to 128 items at three lengths: a short clip, a typical clip and the
longest clip the server takes (`batch.max_item_s` in the manifest, or `batch.lengths_s` to choose them). A server that batches inside one item, like faster-whisper with the 30 s windows of a file, sets `batch.max_n: 1`, so the sweep sends single items of growing length.
Each length stops at the first out of memory error, or when the server had to split a batch, or when a
batch takes more than 120 s. It also stops before a batch it predicts to need more than the budget, because under WSL2 such a
batch does not fail: Windows moves GPU memory to system RAM and the batch takes minutes. No request
carries more than 6,000 s of audio, in the sweep or in a recommendation, because OIP sends samples as
JSON numbers and a larger request can exhaust the server's host memory. It then fits

    peak GiB ≈ base + a × N × L + b × N × L²

where N is the number of items and L is the length of each in seconds. The L² term matters for attention
models. The recommendation for V GB is the largest N whose predicted peak stays under 0.9 × V. Peak
memory is read from `nvidia-smi` during each request, less what the GPU used before the model started.
This is correct because only one GPU model runs at a time. For a PyTorch image it is also read from the
image's own counters (`GET /platform/memory`), and the larger value is kept. CTranslate2 and ONNX images
have only the first reading.

When a build has a profile, each start passes limits for the memory the start was given:
`AISVC_MAX_BATCH` (the batch for the shortest measured length) and `AISVC_MAX_BATCH_LEN` (the padded
length, which is items times the longest item, in milliseconds of audio). Without a profile the image
uses its own small defaults, and its out of memory backoff splits a batch that does not fit.

A vLLM service is not measured, because vLLM batches by itself. Its profile comes from its startup log
(the available KV cache memory and the KV cache size in tokens), read from the whole container log, and
from the last start when the container restarted. The recommendation is the number of sequences of a
given length that fit the KV cache, which is a value for `--max-num-seqs`. The memory outside the KV cache
is the memory its start was given minus the KV cache, so a vLLM service started outside the catalog (or
before the catalog restarted) is not profiled: restart it from the catalog first.

`just batch NAME [--vram GB] [--item-s S | --item-tokens T] [--n N]` prints the same numbers in the
terminal, and `just batch-profile NAME [--wait]` asks for a measurement. The model's page has a Batching
section with the table and a Measure button.

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
 "state": "queued", "position": 2, "waiting_for": "one GPU model runs at a time: pyannote-diarization waits for qwen3-32b-awq (leased by label:job-3)",
 "error": null, "created_at": "…", "granted_at": null, "ready_at": null, "expires_at": "…"}
```

- `state`: `queued` -> `starting` (granted; its models are being started) -> `active` (every model is
  healthy and has a build key: call it now, and index what it returns) -> `released` | `expired` | `failed` (a model failed to start, or the request
  can never be met, `error` says why).
- **Heartbeat** every 30 s; a lease (queued or not) expires after `ttl_s` (default 90 s, i.e. three
  missed beats) without one, so a crashed app frees its models and its queue place.
- One request names its whole model set, with at most one GPU model. A request for a GPU model while
  the same holder leases a different GPU model fails at once, because it would wait forever. Release
  the old lease first.
- `aicore.catalog.Catalog.lease(...)` does all of this (heartbeat thread, waiting, release).

### Scheduler rules

- Only one GPU model runs at a time, and it gets the whole GPU. A queued request for a GPU model is
  granted when no other GPU model is running, or when the one that is running may be stopped. CPU
  models and hosted endpoints run next to the GPU model and never wait for it.
- A request that names two or more GPU models fails at once with HTTP 422. Lease one GPU model per
  stage of a job, together with the CPU models and hosted endpoints that stage needs.
- The scheduler stops a GPU model to make room only when nobody leases it, no call to it is in flight,
  and it has run for its settle time (2 x cold start, at least 60 s) or its lease has ended. If any
  other GPU model can't be stopped yet, the request waits, and `waiting_for` names the model it waits
  for and the reason.
- One model starts at a time, and no GPU model starts while another GPU model is still loading. vLLM
  sizes its KV cache from the free memory it measures at start, so a neighbour that loads at the same
  time breaks its start ("No available memory for the cache blocks").
- `gpu.util` (engines) and `gpu.vram_gb` (custom services) are the least memory a model needs to
  load. The catalog reads free memory from `nvidia-smi` (the container gets the NVIDIA `utility`
  capability for this), so memory used by other processes counts. A GPU model starts only when that
  minimum plus 1 GiB is free. Just before the start the catalog measures again, waiting up to 30 s
  for a stopped model to give its memory back. If memory is still short, the request goes back to its
  place in the queue, and the model is not started at all.
- At start, a GPU model gets `min(util_limit, (free - 1 GiB) / total)` of the GPU, from that last
  measurement. vLLM gets it as `--gpu-memory-utilization`, and custom services get it in GiB as
  `AISVC_VRAM_GB`. The 1 GiB stays free because Windows keeps some GPU memory for the display under
  WSL2, and vLLM fails to start when the memory it asks for isn't free. Without `nvidia-smi`, a GPU
  model gets `util_limit` x `gpu_total_gb`.
- Stopping one GPU model to start a different one counts as a swap. `GET /api/platform` returns the
  count as `swaps`, and the catalog keeps it in its database across restarts.
- **Models a lease brought up are spun down when nothing needs them.** Once its last lease ends, a model
  is stopped after `CATALOG_RELEASE_LINGER_S` (default 30 s, so a back-to-back lease on the same model
  keeps it loaded) unless a live or queued request names it or a call is in flight. A request that
  comes after the stop has begun waits until the model has stopped and started again. Models started on
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
  the outer loop and stays up; the other stage's models are re-spun inside it unless at most one model of the two stages uses the GPU
  beside it, in which case nothing is re-spun.
- `keys` per model are optional. With keys, the catalog checks the index for the current build, and
  a model with nothing missing is skipped (`skip: true`). Keys for a later stage are usually unknown
  until the earlier stage has run (segments decide the cuts); plan again after it.

## Events

`GET /api/events` is a server-sent event stream: `model` events (`{id, state, build_key,
capabilities_status}`), `lease` events (the lease JSON), `queue` events (`{queue: [{id, holder,
position, waiting_for}]}`), `export` events (the export job JSON) and `dataset` events (`{id, n,
action}`, see [Dataset Catalog](#dataset-catalog)). App backends can relay these or poll `GET /api/models` and
`GET /api/leases/{id}`. Each connection starts with a `model` event per model. A client that falls more than
1000 events behind gets no more events: its stream ends once it has read the ones already queued, and a
reconnect (EventSource does this after 3 s) starts again with that snapshot.

## UI

`http://127.0.0.1:8700/` (AeroAudio design system): Models (`?group=none` turns off grouping by model
type), a model's page at `/models/{id}` (apps link there; hosted models get their own layout), Add model
(`/add?name={id}`, `/add?source=hosted`), Export (`/export?model={id}`), Queue and leases, Output index,
Model contract, Docs. Live through `/api/events`. A switch at the top of the Nav moves between the
Model Catalog and the Dataset Catalog, whose pages are under `/datasets` (see [Pages](#pages)).

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
cat.batch("wavlm-sv", item_s=20)["recommendation"]  # {"batch": ..., "vram_gb": ...} once a profile exists
```

A `Catalog` keeps its HTTP connections open until `cat.close()`; `with Catalog() as cat:` closes them at the
end of the block.

`aicore.catalog.FakeCatalog` is an in-memory stand-in with the same methods for app tests (leases are
granted at once; models and capabilities come from the constructor).

## Dataset Catalog

The Dataset Catalog keeps versioned datasets that Concord Bench and Concord Label both read. It runs in the
same catalog service as the Model Catalog, on the same port, and its code is in `catalog/datasets/`. The
design is in [../DATASET_CATALOG_PLAN.md](../DATASET_CATALOG_PLAN.md), and the version format is in
[../contracts/datasets/version.md](../contracts/datasets/version.md).

- A **dataset** has a name, a language, a description, tags and numbered versions. Its id is a 32 character
  hex string.
- A **version** records which audio files the dataset has, plus its references and labels. It is a
  reference package without `audio/` and `outputs/`. A version never changes. Publishing the same content
  as the latest version again returns that version, so a double click doesn't make two versions.
- **Audio** is stored once, under the sha256 of the original bytes, and both apps read it from the same
  folder. No audio is copied between the apps.
- **Only the catalog service writes** to the store. The apps read audio and version folders straight from
  the store. They never write to it.

### The store

All of the Dataset Catalog's state is in one folder, the store:

```
${DATASET_STORE}/
  datasets.db                     SQLite: datasets, versions, audio, holds and uses
  audio/<sha256[0:2]>/<sha256>    the original audio bytes, with no extension
  versions/<dataset_id>/<n>/      one folder per version
  tmp/                            uploads in progress
```

- **Where it is.** The folder is `DATASET_STORE`. When that isn't set, it is `${AI_DATA}/datasets`, and
  `AI_DATA` defaults to `~/ai-data`. Set either one in `.env` (see `.env.example`). The apps and
  `aicore.datasets` use the same default.
- **In Docker.** `just catalog-up` works out the path, creates the folder, and mounts it read-write in the
  container at the same path as on the host. A path the catalog reports is then the same path on the host.
  Bench and Label containers mount the same folder read-only, and they set `DATASET_STORE` to the path they
  see. `just catalog-serve` and apps run on the host read the host folder directly.
- **Audio files.** A file in `audio/` has no extension, because two uploads of the same bytes can arrive as
  `.wav` and `.WAV`. `datasets.db` keeps each file's extension, size, duration, sample rate and channels.
  The catalog reads those with ffprobe when a file is uploaded, so the catalog image includes ffmpeg.
- **Backup.** The records in the store can't be rebuilt, unlike `run/catalog/catalog.db`. That is why the
  store lives outside `PLATFORM_DIR`, in its own folder. Back up the whole folder as one unit.
  `datasets.db` runs in SQLite's WAL mode, so stop the catalog service (`just catalog-down`) before you
  copy the folder. The database and the audio then match.

### API

Every path is under `/api`. Errors have the same shape as the rest of the catalog, `{"error": {"message",
"type", "code"}}`, and some add fields such as `latest`, `missing` or `problems`.

| Call | Does |
|---|---|
| `POST /api/store/missing` | `{"sha256": [...]}` -> `{"missing": [...]}`, the files the store doesn't have. A file that is still uploading counts as missing. |
| `PUT /api/store/audio/{sha256}?ext=&offset=` | Takes raw bytes from `offset` -> `{"size"}`. 409 with the current `size` when `offset` is wrong, so an upload can resume. |
| `POST /api/store/audio/{sha256}/commit` | Checks the sha256 and stores the file -> its audio record. 422 when the bytes don't match, and the partial file is removed. |
| `GET /api/store/audio/{sha256}` | The original bytes, for the UI's player. |
| `POST /api/store/unpack` | `{"sha256"}` of a committed zip -> the audio files in it, each stored and probed, and the files it skipped. |
| `POST /api/store/holds` | `{"app", "add": [...], "remove": [...]}` -> `{"held"}`, the number of files this app holds. |
| `GET /api/store/stats` | Sizes and counts for the Storage page, holds per app and free disk space. |
| `POST /api/store/gc` | `{"dry_run": true}` -> what clean-up would delete. The same call with `false` deletes it. |
| `GET /api/datasets` | `?q=&language=&label_type=&app=` -> datasets, each with its latest version. |
| `POST /api/datasets` | `{"name", "language"?, "description"?, "tags"?, "app"}` -> 201 dataset. |
| `GET`, `PATCH`, `DELETE /api/datasets/{id}` | The dataset with its versions, newest first. PATCH changes the name, description and tags. DELETE removes the dataset and every version. |
| `POST /api/datasets/{id}/versions` | Multipart: a zip of a version folder (or of a whole reference package), plus `parent`, `app` and `force`. 201 for a new version, 200 when it matches the latest version, 409 `stale_parent` with `latest`, 422 when the folder is invalid or audio is missing. |
| `POST /api/datasets/{id}/versions/audio` | `{"files": [{"path", "sha256"}], "parent"}` -> a version with audio only. 409 when the latest version has references. |
| `GET`, `DELETE /api/datasets/{id}/versions/{n}` | The version with its manifest and counts. DELETE removes it, 204. |
| `GET /api/datasets/{id}/versions/{n}/files` | `?offset=&limit=` -> the files, with their segment counts and labels. |
| `GET /api/datasets/{id}/versions/{n}/refs?path=` | The reference segments of one file. |
| `GET /api/datasets/{id}/versions/{n}/package?audio=` | The version as a reference package zip, with or without `audio/`. |
| `POST`, `DELETE /api/datasets/{id}/versions/{n}/uses` | Records `{"app", "ref"}`, an app's dataset made from this version, or removes it with `?app=&ref=`. 204. |
| `GET /api/datasets/{id}/compare?a=&b=` | Files added and removed, and changes to references and labels between two versions. |
| `POST /api/datasets/import` | Multipart reference package zip, plus `dataset_id`?, `app`? and `preview`?. With `preview` it stores nothing and returns what an import would do. |

A version's number is never used again after it is deleted. A later version whose parent was deleted shows
`parent_deleted`. A publish whose `parent` isn't the latest version gets a 409, and the app asks the person
whether to publish anyway with `force`. Each change sends a `dataset` event on `/api/events`, with
`action` set to `created`, `updated`, `published` or `deleted`. `n` is null when the event is about the
whole dataset.

### Pages

The UI's Dataset Catalog tab has these pages. The Storage widget in the Nav takes the place of the GPU
widget while the tab is open.

| Page | Path | Shows and does |
|---|---|---|
| Datasets | `/datasets` | Every dataset with its latest version, files, hours and label types. Search, and filter by language, label type and app. |
| Dataset | `/datasets/:id` | The version history, with the apps' datasets made from each version. Edit the name, description and tags, download a version with or without audio, add audio, and delete a version or the dataset. |
| Version | `/datasets/:id/v/:n` | The files with a player, one file's references, and label statistics. "Open in Label" opens the same file in Concord Label. |
| Compare | `/datasets/:id/compare?a=&b=` | What changed between two versions, with a text diff for each edited segment. |
| Upload audio | `/datasets/upload` | Takes files, a folder or zips of audio without references. The browser hashes each file, sends only what the store lacks in 16 MB chunks, and resumes after a dropped connection. It then makes version 1 of a new dataset, or adds the files to a dataset whose latest version has audio only. |
| Import package | `/datasets/import` | Previews a reference package zip, then imports it as a new dataset or a new version of one. |
| Storage | `/datasets/storage` | The store's size, audio that is in versions, held only, or in neither, and holds per app. Runs clean-up. |

"Open in Label" shows only when Concord Label has registered with the catalog (`POST /api/apps`) at an
address a browser can reach, such as `localhost` or a name with a dot in it. A Label container registers
under its name on `ai-net` (`http://concord-label:8620`), which a browser can't open, so the button is
hidden in that case.

### Holds and clean-up

Audio is uploaded before any version lists it. Label, for example, stores audio when a person uploads it,
and the person may label it for weeks before they publish. So each app places a **hold** on every file one
of its datasets keeps, and it releases the hold when it deletes that dataset. A dataset an app made from a
version holds its audio too. Deleting a version therefore never removes audio an app still reads.

Clean-up deletes an audio file only when all of these are true:

- No version lists it.
- No app holds it.
- It has been in the store for more than 24 hours. For an upload that stopped partway, the 24 hours count
  from its last write.

Clean-up runs only when a person starts it, and it always shows a dry run first. Start it from the Storage
page, or run `just datasets-gc`. The recipe asks the running catalog service for a dry run, prints the
files and bytes it would delete, and deletes them only when you answer yes. Deleting a version or a
dataset never deletes audio by itself.

### Python client (aicore.datasets)

`aicore.datasets.DatasetCatalog` calls the Dataset Catalog's API on the same catalog service, so it uses the
same URL as `Catalog`. Each app passes its own name as `app`, and the client sends that name with every dataset
it creates, every version it publishes, and every hold and use it records. The apps read audio and versions from
the mounted store, which is `DATASET_STORE`, or `${AI_DATA}/datasets` when that isn't set, or
`~/ai-data/datasets`. The catalog service uses the same default.

```python
from aicore.datasets import DatasetCatalog, MissingAudio, StaleParent

dc = DatasetCatalog(app="concord-label")            # AICORE_CATALOG, else http://127.0.0.1:8700
sha = dc.upload_audio("call.wav")                   # skips files the store already has
dc.hold([sha])                                      # dc.release([sha]) when the app no longer needs it
d = dc.create_dataset("Help desk calls", language="en")
try:
    v = dc.publish(d["id"], "out/version", parent=None)   # a version folder (contracts/datasets/version.md)
except StaleParent as exc:
    ...                                             # exc.latest is the latest version; publish again with force=True
except MissingAudio as exc:
    ...                                             # exc.missing lists the sha256s to upload first
dc.version_dir(d["id"], v["n"])                     # the version's folder in the mounted store
dc.audio_path(sha)                                  # the original audio in the mounted store
dc.add_use(d["id"], v["n"], ref="12")               # the app made its dataset 12 from this version
```

`upload_audio` sends a file in 16 MB chunks and then commits it. When the connection drops, or when an earlier
upload of the same file stopped partway, the store answers with the size it already has, and the client goes on
from there. `publish` returns the version with `created` set to false when the folder has the same content as
the latest version, and the catalog returns that version instead of making a new one. The other calls are
`datasets`, `dataset`, `update_dataset`, `delete_dataset`, `version`, `files`, `publish_audio`,
`delete_version`, `remove_use` and `missing`. Every error is a `DatasetCatalogError`, which is a
`CatalogError`, and `NotFound`, `StaleParent`, `HasReferences`, `MissingAudio` and `InvalidVersion` are its
subclasses for the cases an app handles.

`aicore.datasets.FakeDatasetCatalog` has the same methods for app tests. It keeps its records in memory, and it
writes audio and versions to a folder laid out like the real store, so `audio_path` and `version_dir` work. It
follows the same rules as the catalog service for publishing, holds and uses. Pass a folder such as pytest's
`tmp_path` as `root`, or leave it out to get a temporary folder that is removed with the fake. A test can put
audio straight into the store with `add_audio(data)`, and it can publish a version with `add_version(dataset_id,
{path: bytes}, {refs file name: text})`. The fake reads the duration and sample rate of WAV files only, so pass
`duration` and `sample_rate` to `add_audio` for other formats. `calls` lists every call that changed something,
and `held()` returns the sha256s the app holds. `catalog/tests/test_dataset_client.py` runs the same tests
against the fake and against the real client, so the two stay in step.
