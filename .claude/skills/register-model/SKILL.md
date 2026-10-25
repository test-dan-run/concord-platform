---
name: register-model
description: Register a new model on the personal AI service platform (services/*.yaml) so it can be started with `just up` and called through aicore, or register a model a provider already serves online as a hosted endpoint in the Model Catalog. Use when a user or agent wants to add, register, onboard or serve a model (a Hugging Face id, a local checkpoint in $AI_ASSETS, a model name like "Llama 3.1 8B", or a provider URL and API key), give it a port/alias/profile, or work out its gpu.util. Covers choosing the API contract (including speech enhancement and text.itn), engine vs custom server, VRAM sizing, writing the manifest, batching (vLLM --max-num-seqs, aisvc batchers and the manifest batch hints in custom servers, measuring a batch profile), audio longer than a model's window, updating the pinned tests and README, validating, and catalog registration and verification.
---

# Register a model on the platform

This skill lives in the platform repository (`.claude/skills/register-model/`); run every command
below from the platform root. One YAML manifest per service in `services/` is the source of truth;
`just render` generates compose, the aicore endpoint map and the catalog from it. Registering a model
means writing that manifest correctly, sizing its VRAM, and keeping the tests and README in step.
`../README.md` (the platform guide, one level up) and `contracts/README.md` hold the full rules; this
skill is the procedure.

Two helpers ship with this skill (run them with the platform venv):

```bash
S=.claude/skills/register-model/scripts
uv run --quiet python $S/platform_state.py --api chat.completions
uv run --quiet python $S/size_model.py Qwen/Qwen3-8B-FP8 --max-model-len 8192
```

- `platform_state.py`: engines and pinned tags, every service with alias/port/source, used ports
  and the **next free port** per band, and the GPU budget (`util_limit`).
- `size_model.py SOURCE`: reads `config.json` and file sizes (local dir, `/models/<dir>`, or an HF id,
  metadata only) and prints architecture, quantization, checkpoint GB, **KV bytes per token**, and a
  `gpu.util` for a given `--max-model-len` (or, with `--util`, how many KV tokens fit).

## Guardrails

- **Don't download weights or pull engine images** unless the user asks. Writing and validating a
  manifest needs neither. `just up` pulls both, so confirm before running it.
- **Never reassign an existing alias** (`default-chat` etc.) unless the user asks: apps resolve
  aliases at runtime, so this silently changes which model they call.
- **Don't change an existing profile's members** (that changes what `just up <profile>` starts)
  unless the user named that profile. Default to a new profile named for the role.
- Never `:latest`, never weights in images or git, ports bind to `127.0.0.1` (render does this).
- When you can't confirm an engine flag or capability for the pinned version, write
  `# TODO(verify): <what to check>` in the manifest and list it in your report. Don't guess silently.

Ask the user only for decisions that are theirs: taking over an alias, joining an existing profile,
going over budget, downloading, or escalating past a stock engine. Derive everything else.

## Procedure

### 0. Platform service or hosted endpoint?

If a provider already serves the model online (a base URL and an API key) and the user wants it
called from here rather than run here, it is a **hosted endpoint**: no manifest, port, VRAM or image.
It is registered in the Model Catalog only. Follow
[references/hosted-endpoint.md](references/hosted-endpoint.md) instead of the steps below. Everything
else is a platform service: continue at step 1.

### 1. Check it isn't already registered

```bash
grep -rn -i "<repo-or-dir-fragment>" services/
```

If a manifest already serves the same checkpoint, stop and report it. A different quant or
context of the same model is a new service only if the user wants both.

### 2. Pick the API (and so the contract)

| The model does | `api` | `contract` | Port band |
|---|---|---|---|
| Chat / instruct text generation, VLMs (images/video in content parts) | `chat.completions` | `openai` | 82xx |
| Raw completion (base models) | `completions` | `openai` | 82xx |
| Embeddings | `embeddings` | `openai` | 83xx |
| Speech to text | `audio.transcriptions` | `openai` | 81xx |
| Text to speech | `audio.speech` | `openai` | 81xx |
| Non-text input, no OpenAI route (VAD, diarization, classifiers...) | `infer` | `oip-v2` | 84xx |
| Speech enhancement: audio in, audio out (denoising, dereverberation, separation to one target) | `infer` | `oip-v2` | 84xx (85xx as a utility) |
| Inverse text normalisation: spoken form to written form ("three thirty p m" → "3:30pm"), with spans | `text.itn` | `openai-inspired` | 85xx (utility) or 84xx |
| Other text in, text out, no OpenAI route | `text.transform` | `openai-inspired` | 85xx (utility) or 84xx |

The first matching row wins (the order is `openai` → `oip-v2` → `openai-inspired`). An ITN model
is `text.itn`, never `text.transform`: `text.itn` has its own request (`input` string or list,
scalar `options` `language`, `return_spans`, comma-separated `classes`), response (`data[].output`,
`data[].spans` with half-open code-point offsets) and seven conformance checks; see the `text.itn`
section of `contracts/README.md` and `contracts/openapi/text.itn.yaml`. A provider's ITN API that
does not follow it (a top-level `language`, array options, ignored unknown options) does not conform.

An `oip-v2` signature decides which capability probe the catalog runs, so use the shared shapes when
the task matches (custom servers: [references/custom-model.md](references/custom-model.md)):

| Task | Inputs | Outputs | Catalog verifies it as |
|---|---|---|---|
| VAD | `AUDIO` FP32 [-1], `SAMPLE_RATE` INT32 [1] | `SEGMENTS` FP32 [-1, 2] | VAD |
| Diarization | `AUDIO` FP32 [-1, -1] (zero-padded rows), `LENGTHS` INT32 [-1], `SAMPLE_RATE` INT32 [1] | `SEGMENTS` FP32 [-1, 2], `SPEAKERS` INT32 [-1], `ROWS` INT32 [-1] (the row of each segment) | diarisation |
| Speech enhancement | the same as diarization | `AUDIO` FP32 [-1, -1] (zero-padded), `LENGTHS` INT32 [-1], `SAMPLE_RATE` INT32 [1] (each row keeps its rate and length) | speech enhancement |

The diarization and enhancement servers also take one clip as `AUDIO` [-1] without `LENGTHS` and answer
it in the single-clip shapes, because the apps still send that form. A new server of either kind copies
`server/rows.py` from `nemotronlabs-re-use` for that, and the copies must stay identical
(`tests/test_images.py`). The catalog detects both forms by tensor names.

Give `SAMPLE_RATE` `example: [16000]`. Tensors travel as JSON number arrays; the binary tensor
extension is not implemented. For a dereverberation model, put `dereverberation` in `tasks`: the
probe then uses a room response instead of babble noise.

If nothing
fits (rerank, for example, has no API type yet), **stop**: a new API type needs a spec in
`contracts/README.md`, an entry in `contracts/api_types.yaml` and a conformance suite first. Report
that to the user instead of forcing the model into a wrong `api`.

### 3. Pick the rung: stock engine first

Climb only as far as needed:

1. **Stock engine** (`engine: vllm`, or `sglang`). Requirements:
   - The checkpoint is HF format (`config.json` + safetensors/bin). GGUF, CTranslate2, ONNX and
     `.nemo` are not; see `faster-whisper-large-v3` and `nemotron-3-diarization`.
   - The architecture (from `size_model.py`) is supported by the **pinned** tag in
     `engines/engines.yaml`. Check the supported-models page for that exact version
     (`https://docs.vllm.ai/en/<tag>/models/supported_models.html`). If the image is already
     local (`docker image inspect <image>` succeeds), you can ask the registry directly:
     `docker run --rm --entrypoint python3 <image> -c "from vllm import ModelRegistry as R; print('<Arch>' in R.get_supported_archs())"`.
   - The quantization runs on the RTX 5090 (sm_120). FP8, AWQ and GPTQ are in use here already.
     Anything else (MXFP4, NVFP4, bitsandbytes, ...) gets a `TODO(verify)` unless you confirmed it.
   - Prefer vLLM. Use SGLang only when vLLM can't serve the model and SGLang can; its pinned tag has
     not been run on the 5090 yet (see the TODO in `engines.yaml`), so say so in the report.
2. **Engine plus plugin**: a Dockerfile `FROM` the pinned engine image plus a new `engines.yaml`
   entry. This changes shared infrastructure, so confirm with the user first.
3. **Custom server** on a shared base: read [references/custom-model.md](references/custom-model.md).
4. **Fully bespoke** (`build.base: none`): last resort, same contract and conformance rules.

Steps 4–7 below are written for rung 1. A custom model follows the reference file for files and
fields, then rejoins at step 6 (tests/README) and step 7 (verify).

### 4. Decide the weight source and size it

- **Local copy already in `$AI_ASSETS`** (path in `.env`; `ls $AI_ASSETS`): use
  `model: /models/<dir>` (mounted read-only) and add a provenance comment from its `SOURCE.txt`:
  `# Local copy of <repo> @<first 7 of revision> in ${AI_ASSETS}/<dir>.`
- **Otherwise download a local copy first.** Validation rejects Hugging Face ids (`model: Org/Repo`):
  every engine model is a local copy, and services start with `HF_HUB_OFFLINE=1`. Pin the current
  commit and download it:
  `uvx --from huggingface_hub hf download <repo> --revision <sha> --local-dir $AI_ASSETS/<dir>`, then
  delete `$AI_ASSETS/<dir>/.cache` and write `SOURCE.txt` in the same format as its neighbours (`repo`,
  `revision`, `source`, `downloaded`). Use `model: /models/<dir>` with the provenance comment above.
  Gated repos need `HF_TOKEN` in `.env` for the download: tell the user.

Size it:

```bash
uv run --quiet python $S/size_model.py <source> --max-model-len <N> [--overhead-gb 3 for VLMs]
```

- **Context.** Use the native context if it fits. Otherwise cap it with `--max-model-len` in `args`
  and a comment giving the native length. Chat models here usually run 8k–32k.
- **Overhead.** The default of 1.5 GB fits text models up to about 14B. Use about 3 GB for VLMs
  (vision encoder + encoder cache). Embedding/pooling models need no KV cache.
- **util** = the suggestion rounded up to 0.01. Weights over about 0.9 × 32 GB do not fit: say the
  model needs a smaller quant rather than writing a manifest that cannot start.
- **What util and vram_gb mean.** `gpu.util` (custom: `gpu.vram_gb`) is the least memory the model
  needs to load, and it must stay at or below `util_limit` (0.93). Only one GPU service runs at a
  time, and when it starts it gets `min(util_limit, (free - 1 GiB) / total)` of the GPU, measured just
  before the start. Size the minimum from the arithmetic or a measurement, because validation, bundles
  for smaller GPUs and the start check all use it. Don't add room for other services, because GPU
  models never share the GPU.

### 5. Choose identity and write the manifest

`platform_state.py --api <api>` gives the next free port and the budget.

- **name**: kebab-case from the repo name, lowercased, without the org, `.` → `-`, and the quant
  suffix kept when it distinguishes the checkpoint (`qwen3-32b-awq`, `qwen2-5-14b-instruct-awq`). The
  file is `services/<name>.yaml`; the name is also the served model name.
- **port**: the next free port in the band from step 2. A port outside its band is a warning, and
  `just test` asserts there are no warnings, so it fails the tests.
- **alias**: only when the user asks, or when the model takes a new role (`fast-chat`, `mid-chat`,
  `default-vlm`). It must be unique across names and aliases.
- **profiles**: a new kebab-case profile for the role, or the one the user named.

Template (match the comment style of the existing manifests: the arithmetic goes on the `util` line
and the reason for each flag goes on its `args` line):

```yaml
name: <name>                       # (comment when the name differs from the repo, e.g. "2.5" -> "2-5")
kind: model
alias: <alias>                     # omit when none
engine: vllm
contract: openai
api: chat.completions
# Local copy of <Org/Repo> @<sha7> in ${AI_ASSETS}/<dir>.
model: /models/<dir>               # always a local copy; HF ids fail validation
port: 82NN
gpu:
  util: 0.NN                       # NN.N GB: X.X GB <quant> checkpoint + ~Y GB KV (~Zk tokens at K KB/token)
args:
  - --max-model-len=NNNNN          # native NNNk; capped to fit the KV cache in the budget
profiles: [<profile>]
description: "<Role>, <Model> <size> <quant>; <notable behaviour>"
tasks: [<task>, ...]               # kebab-case, from the model card: what it was trained for, not only what this service exposes
```

Flags for the pinned vLLM. Verify each one against that version's docs before you use it:

- Thinking models: `--reasoning-parser=<family>` (e.g. `qwen3`). Note in `description` how to turn
  thinking off (`chat_template_kwargs`), as `qwen3-8b` does.
- VLMs: `'--limit-mm-per-prompt={"image": N, "video": M}'` (bounds encoder memory at profiling).
- Embeddings: `--runner=pooling` (v0.31 replaced `--task embed`).
- Whisper-style ASR: `--max-model-len=448`.
- `--max-num-seqs`: how many sequences vLLM runs at once. The service has the whole GPU, so don't keep
  a small value (4) that was sized for a shared GPU. Apps send work in concurrent batches
  (`aicore.batching`), so give a model that serves short requests room for them, such as 64 for ASR, as
  `qwen3-asr-1-7b` and `granite-speech-5-0-2b` do. Boot it at its minimum `util` and at 0.90, and
  record the KV cache size of each boot on the `args` line. Keep a smaller value only for a reason
  you can state (`qwen-sea-lion-v4-5-27b-it` bounds its state memory at 16). The catalog's batch view
  suggests a value from the startup log (step 7).
- Tool calling, only if the user wants it: `--enable-auto-tool-choice --tool-call-parser=<family>`.

### 6. Update the pinned tests and the README

`just test` pins the registry, so a new service makes it fail until you update these:

- `tests/test_schema.py::test_sample_manifests_validate`:
  - Add the name to the sorted `names` list.
  - Add any new profile to the sorted `profiles` list.
  - Add `"<alias>": "<name>"` to the alias dict. With no alias, note that the `None:` key holds the
    alphabetically last alias-less service, because it is a dict comprehension. Update it if your
    name sorts after the current one.
- `tests/test_stack.py`: only if the service joins `voice-chat` (the expected start order and
  `stop` tuples are hard-coded there).
- `README.md` (platform root): add a row to the table of the service's group, e.g. "Speech to text
  (ASR)" or "Text normalisation" (name, alias, kind, engine, contract / api, port, `util 0.NN` or
  `N GB` or `CPU`), in port order. A service that fits no group goes in "Other audio analysis" or a
  new group with a one-sentence description.

### 7. Verify

```bash
just validate          # must print "ok" with no warnings
just vram <name>       # must pass (exit 0); a set with two GPU services fails
just render
just test
```

All four must pass before you report success. Stop there unless the user asked for a live start
or for the model in the catalog.
If they did:

```bash
just up <name>         # pulls image/weights on first run; one service at a time, health-gated
just logs <name>       # look for vLLM's "GPU KV cache size: N tokens" and "Maximum concurrency"
just conform <name>
```

If startup fails with no memory for cache blocks, lower `--max-model-len` before raising `util`. Once it
runs, replace the estimate in the `util` comment with what was measured, as `faster-whisper-large-v3`
does (`measured 4.1 GB loaded`).

**Catalog.** Apps list only models registered in the Model Catalog (`catalog/README.md`,
http://127.0.0.1:8700). When the user wants the model listed, and the catalog is running
(`curl -s localhost:8700/api/health`), register it; the catalog picks up the new manifest by itself:

```bash
curl -s -X POST localhost:8700/api/models/<name>/register -H 'content-type: application/json' \
  -d '{"tags": ["<tag>"], "display_name": "<optional>"}'
curl -s localhost:8700/api/models/<name>        # capabilities.status: pending -> running -> verified | failed
```

Registration queues verification, which takes a lease and may start the model (confirm first, as for
`just up`). It runs the conformance checks, then the probe for the API: ASR transcribes the check clip
(timing level, WER); a segmenter is told VAD from diarisation by `SPEAKERS`; an enhancer gets the clip
with babble at 5 dB SNR (or a room response for `dereverberation`) and fails on a wrong rate, a length
more than 10 ms off, NaN or clipping, then passes when SI-SDR improves and warns when it does not;
`text.itn` gets spoken-form phrases and passes when every output holds the expected digits
("twenty one" → 21, "three thirty p m" → 3:30). Read `capabilities.report.lines` and report them.

**Batch profile.** The catalog does not profile a new GPU model by itself. Once it is registered and
verified, ask for a profile (it leases the model and may start it, so confirm first, as for `just up`)
and read the result:

```bash
just batch-profile <name> --wait   # custom image: a sweep; vLLM: the KV cache size from the startup log
just batch <name>                  # the recommendation, and for a custom image the limits the next start gets
```

For a custom image, the sweep sends batches of 1, 2, 4, ... at a short, a typical and the longest
length, and each later start gets `AISVC_MAX_BATCH` and `AISVC_MAX_BATCH_LEN` for the memory it was
given. Without a profile, the image uses its own small defaults. If the sweep stops early or the
limits look wrong, check the manifest's `batch` hints
([references/custom-model.md](references/custom-model.md#batching)). A person can override the limits
on the model's Batching panel. For vLLM, `just batch <name> --item-tokens <T>` gives a
`--max-num-seqs` value for requests of T tokens. A CPU service has no profile, and an API with no
batch measurement fails the profile with a message saying so: report that, it is not an error in the
manifest.

### 8. Report

Tell the user:

- The manifest path, plus the contract, api, engine, port, alias and profile, with the reason for
  any non-obvious choice.
- The sizing arithmetic, that is, the minimum the model needs to load.
- The output of validate, vram and test, and whether a live start and conformance ran. If they
  didn't, give the exact `just up` / `just conform` commands.
- Every `TODO(verify)` you left, and anything the user must do (HF license or `HF_TOKEN`,
  downloads).
- Whether it was registered in the catalog, and its verification result.
- Batching: the `--max-num-seqs` value and why, or for a custom GPU image the `batch` hints and the
  measured limits (or the `just batch-profile <name> --wait` command, when it was not measured).
- How to call it: `client("<alias-or-name>")` with `model=model_name("<alias-or-name>")`. For
  `text.itn`: `client("<name>").itn("meet at three thirty p m", return_spans=True)`. For `oip-v2`:
  `client("<name>").infer({"AUDIO": audio, "SAMPLE_RATE": np.array([16000], dtype=np.int32)})`.
