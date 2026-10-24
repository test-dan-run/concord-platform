# Model catalog — design

Oct 22, 2026 · @Dan

**Status:** implemented in `platform/catalog/` (contract: `platform/catalog/README.md`). Decisions taken
while building it: index lookups are explicit catalog calls from each app's cache layer, not inside the
gateway; the gateway is addressed by a per-model path prefix (`/m/{id}/...`); outputs found in the other
app are copied into the reader's own store; engine models are always local copies in `$AI_ASSETS`, so the
build key covers their weights. The catalog listens on 8700.

## Purpose and scope

One catalog service registers, verifies and runs every model service, so Benchmark and Labelling never load or verify a model themselves. Both apps ask the catalog for models by capability and send their inference calls through it.

- **In scope:** reading model definitions, verifying contracts, recording capabilities, starting and stopping model services within the available compute, routing inference calls, indexing model outputs, planning multi-model jobs.
- **Out of scope:** datasets, metrics, labels and training. Those stay in the apps.
- **Where it lives:** the catalog is the control plane of the Personal AI service platform, not a third standalone app. The platform repo's README stays the source of truth for model contracts and per-model config; the catalog reads that config and adds nothing the platform already defines.
- **Deployment:** one catalog per compute environment. It manages only the GPUs of the host it runs on.

## Responsibilities

The catalog has four parts: registry, verifier, scheduler and gateway. They ship as one service, but the scheduler is its own module so it can later be swapped for an off-the-shelf one.

| Part | Does | Reads | Writes |
| --- | --- | --- | --- |
| Registry | Lists every model the platform defines, with its tags | Platform repo per-model config | Model records |
| Verifier | Runs contract checks against a live instance and records what the model can do | Model records, platform contract | Capability records |
| Scheduler | Decides which services run, starts and stops them, plans multi-model jobs | Capability records, footprints, leases, queue | Service state, job plans |
| Gateway | Takes the apps' inference calls, starts models on demand, holds calls while loading, checks the output index | Service state, output index | Usage log, output index |

&#91;embedded content: catalog architecture · 4 parts, 2 apps, model services\]

The apps talk only to the gateway and scheduler; the scheduler starts and stops services using the capabilities and cold-start times the verifier records.

Registering a model stays minimal: pick an entry from the platform registry, add tags. Everything else comes from the platform config or is measured by the verifier.

## Verification, capabilities and output index

Every verification result and every indexed output is keyed to the exact build: the **build key** = image digest + hash of the model's platform config. A rebuilt image or a changed config is a new build, so it is verified again automatically; a model name alone never is.

**Capability record** (one per build):

| Field | Example | Source |
| --- | --- | --- |
| Protocol and routes | OpenAI transcriptions, verbose\_json | Verifier |
| Tasks | ASR · diarisation · speech-only VAD · KWS | Verifier, from outputs |
| Output detail | word timestamps, segment speakers, KWS scores | Verifier |
| Footprint | GPU memory + headroom, GPU count | Platform config (declared) |
| Cold-start time | median of measured loads | Verifier and scheduler (measured) |
| Throughput | real-time factor on a reference clip | Verifier (measured) |
| Status | verified · failed (with reasons) · stale | Verifier |

- The verifier runs on first registration, on a new build, and on demand. A failed check names the missing route or tensor, as Add model does today.
- A model whose footprint exceeds the whole host is rejected at registration.
- **Output index:** each app stores its own model outputs. The catalog keeps only an index: build key + audio fingerprint (sha256) + request parameters → owning app + output id. A hit held by the other app is read from that app, so a model already run on a file in either app is never run again. Diarisation segments recorded in this index are what make stage-by-stage pipelines possible (see Job planning).

## Gateway

Apps never call a model service directly; every inference call goes through the catalog. A direct URL breaks the moment the scheduler stops that model, while the gateway can start it, wait and answer.

- **Same protocols out as in.** The gateway exposes the platform contract unchanged (OpenAI-compatible routes and OIP v2), addressed by model id, so app code is the same as calling a model directly.
- **On-demand start.** A call to a stopped model asks the scheduler to start it; the call is held until the model is ready or the start timeout passes: 2 × measured cold-start time, or 10 minutes for a build not yet measured (the app shows "starting").
- **Index first.** A call whose build key, audio fingerprint and parameters are already in the output index returns a reference to the stored output (owning app + output id) without touching the GPU; the calling app reads it from that app.
- **Usage tracking.** Open calls count as activity: a model with calls in flight is never stopped. Last-used time and per-model call counts feed eviction.
- **Overhead.** One extra local hop per call, negligible next to transcribing audio.

## Scheduler

The scheduler runs as many models as fit in the host's GPU memory. When a new model would not fit, it stops the running models that are cheapest to bring back first, but never one an app has leased.

**Compute accounting**

- Fit is checked against **declared footprints** from the platform config, not live memory readings. vLLM reserves memory up front, so live usage does not predict what the next model needs.
- Free memory = host GPU memory − sum of running footprints − a fixed host reserve. All GPUs on the host are treated as one memory pool; per-GPU placement is out of scope for now.
- For vLLM models, the platform config should set `gpu_memory_utilization` below the default of about 0.9, so a small diariser can run beside an ASR model.

**Rules for starting and stopping**

1. **Lease.** An app takes a lease on a model for the length of a job (a benchmark run, a pseudo-labelling pass). A leased model is never stopped; a conflicting request waits in a queue. Leases are kept alive by a heartbeat every 30 s and expire after 3 missed beats (90 s), so a crashed app frees its models; outputs it already stored stay, and a rerun skips them.
2. **In-flight calls.** A model with calls open through the gateway is never stopped.
3. **Settle time.** A model must run for a minimum time before it can be stopped, so two apps alternating between models do not keep reloading them. Settle time = 2 × the model's measured cold-start time, minimum 60 s.
4. **Cost-aware eviction.** Among models free to stop, stop first the one with the lowest *reload cost* = measured cold-start time × likelihood of reuse (recent use and queued demand). A diariser that loads in seconds goes before a vLLM instance that takes minutes, even if the vLLM one was used less recently.
5. **Stop only enough.** Stop models until the new one fits, never more.

**Queue.** Requests that cannot start yet are queued strictly first come, first served. Neither app has priority over the other. Each queued request reports its position and what it is waiting for, so apps can show "waiting for compute".

**Off-the-shelf parts to evaluate** before writing the start/stop layer: llama-swap (on-demand swapping for OpenAI-compatible servers), Triton explicit model-control mode (OIP models), vLLM sleep mode (frees GPU memory without a full restart). Not yet checked against current versions.

## Job planning

A multi-model job (a benchmark grid, an ensemble pseudo-labelling pass) is planned to minimise total cold-start time: the sum, over every model load in the plan, of that model's measured cold-start time. Cold-start time is the only cost the planner weighs.

**Pipelines run stage by stage.** A diariser's output does not depend on the ASR model after it, so the planner:

1. runs each diariser over the whole dataset, stores its segments in the app, then releases it;
2. runs each ASR model over every stored segment set in one stretch, then releases it.

Each model loads exactly once, which is the lowest possible total cold-start time. For a grid of S segmenters × A ASR models:

| Plan | ASR loads (slow, vLLM) | Diariser loads (fast) |
| --- | --- | --- |
| Stage by stage, stored segments | A | S |
| Grouped by ASR, diarisers re-spun | A | S × A |

For 2 × 2 that is 2 + 2 loads instead of 2 + 4; for 6 × 2, 6 diariser loads instead of 12.

**Fallback: co-resident stages.** When both stages must run at once (streaming pipelines, or new audio added mid-run), the slowest-loading model goes in the outer loop and stays up; cheaper models are re-spun inside it. If both fit in memory together, nothing is re-spun.

**Other planning rules**

- Outputs already in the index are skipped before planning, so a re-run only loads models with missing work.
- Within a stage, the order is whichever needs the least total cold-start time.
- The plan is shown to the app before it runs: models in order, loads expected, estimated load time.

## API sketch

Apps need six catalog calls plus the gateway; paths are a sketch, not final.

| Call | Purpose |
| --- | --- |
| `GET /models?task=asr&tag=call-centre` | List models with capability record and current state (running, stopped, starting, queued) |
| `GET /models/{id}` | One model: capability record, build key, cold-start time, verification report |
| `POST /models/{id}/verify` | Re-run verification on demand |
| `POST /jobs` (models, dataset fingerprints, pipeline shape) | Ask for a plan; returns ordered stages, expected loads and index hits |
| `POST /leases` / `DELETE /leases/{id}` | Take and release a lease for a job; returns queue position while waiting |
| `GET /events` (server-sent events) | Model state changes and queue progress for the apps' status UI |
| Gateway: `/v1/...` and `/v2/models/{id}/infer` | Inference, same contract as the platform, addressed by model id |

## Effect on Benchmark and Labelling

- **Add model and Model service contract move to the catalog.** The apps keep no model registration screens.
- **Model pickers** in both apps list catalog models filtered by task and tag, each showing state (running, stopped, starting, queued), capabilities and cold-start time.
- **Benchmark** submits a grid as a job, shows the plan (stages, loads, index hits) before running, and shows "waiting for compute" with queue position while blocked.
- **Labelling** takes a lease for a pseudo-labelling pass; outputs Benchmark already stored are found through the catalog's index and read from Benchmark.
- **Export from Labelling to Benchmark** is unchanged; datasets stay in the apps.
